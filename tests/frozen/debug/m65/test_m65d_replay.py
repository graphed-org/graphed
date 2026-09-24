"""M65 D frozen suite: capture into an M8 store through ``aggregate_plan(store=)`` and
``graphed.debug.replay`` of one task at opt_level=0 (plan-D D-1..D-6). New API is read through module
attributes so each test fails on its own line before the implementation exists."""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import awkward as ak
import numpy as np

import graphed.checkpoint as gcp
import graphed.debug as gd
from graphed import Array, Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.core import (
    GraphStore,
    LocalResources,
    Partition,
    PayloadDescriptor,
    Plan,
    SequentialRunner,
    Task,
)
from graphed.debug.runner import _stage_error
from graphed.execute import compile_ir, external_key

N = 1000
ROWS = 250
_SLOW: list[int] = []
_DROP: list[int] = []


def _first(v: list[Any]) -> float:
    return float(v[0])


def _first_plus_one(v: list[Any]) -> float:
    return float(v[0]) + 1.0


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _rows(v: list[Any]) -> dict[str, tuple[Any, ...]]:
    return {"rows": (v[0],)}


def _rows_add(a: dict[str, tuple[Any, ...]], b: dict[str, tuple[Any, ...]]) -> dict[str, tuple[Any, ...]]:
    return {"rows": a["rows"] + b["rows"]}


def _rows_empty() -> dict[str, tuple[Any, ...]]:
    return {"rows": ()}


def _fail_on_key1(a: Any) -> Any:
    if float(a[0]) == float(ROWS):
        raise ValueError("row 250")
    return a


def _slow(a: Any) -> Any:
    _SLOW.append(1)
    time.sleep(0.06)
    return a


def _drop_after_first(a: Any) -> Any:
    _DROP.append(1)
    return a if len(_DROP) == 1 else a[:-1]


def _ext_sum(x: Any) -> float:
    return float(ak.sum(x))


def _ext_sum_plus_one(x: Any) -> float:
    return float(ak.sum(x)) + 1.0


def _ext_sum_plus_two(x: Any) -> float:
    return float(ak.sum(x)) + 2.0


def _plain(p: Partition, _r: object) -> float:
    return 1.0


def _run_task(process: Any, partition: Partition) -> Any:
    return process(partition, LocalResources())


@dataclass(frozen=True)
class _ExtForm:
    def describe(self) -> str:
        return "sum"


def _parquet(tmp_path: Path) -> str:
    path = tmp_path / "d.parquet"
    ak.to_parquet(ak.Array({"x": np.arange(float(N))}), path)
    return str(path)


def _events(tmp_path: Path) -> tuple[Session, Array]:
    s = Session(AwkwardBackend())
    return s, from_parquet(s, "events", _parquet(tmp_path))


def _plan(*outputs: Array, **kw: Any) -> Plan[Any]:
    kw.setdefault("reduce", _first)
    kw.setdefault("combine", _add)
    kw.setdefault("empty", _zero)
    return aggregate_plan(*outputs, steps_per_file=4, **kw)


def _task(plan: Plan[Any], key: int) -> Task:
    return next(t for t in plan.tasks if t.key == key)


def _by_stage(done: dict[str, Any], stage: str) -> set[str]:
    return {e.partition for e in done.values() if e.stage == stage}


def _external(s: Session, x: Array) -> tuple[Array, str]:
    desc = PayloadDescriptor(
        kind="histogram",
        content_hash="sha256:m65d",
        framework="boost_histogram",
        version="1",
        io_schema="uhi",
        preprocessing_ref=None,
    )
    h = s.record_external("histogram", _ext_sum, [x], {"spec": "m65d"}, descriptor=desc, form=_ExtForm())
    node = next(
        n for n in GraphStore.deserialize(bytes(compile_ir(s, h).ir)).nodes() if n["kind"] == "external"
    )
    return h, external_key(node)


def test_capture_writes_inputs_and_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    root = tmp_path / "cap"
    plan = _plan(y, store=root)
    SequentialRunner().run(plan)
    store = gcp.Store(str(root))  # a path root is a Store directory, not an fsspec layout
    done = store.completed()
    parts = {str(t.partition) for t in plan.tasks}
    assert len(parts) == 4
    assert _by_stage(done, "replay-input") == parts
    assert _by_stage(done, "replay-output") == parts
    assert len(done) == 8
    assert all(store.get(e.blob) for e in done.values())

    bad = gak.sum(ev.x.map(_fail_on_key1, name="fail") * 2.0, axis=None)
    root1 = tmp_path / "cap1"
    failing = _plan(bad, store=root1)
    with pytest.raises(gd.StageError):
        SequentialRunner().run(failing)
    done1 = gcp.Store(str(root1)).completed()
    assert _by_stage(done1, "replay-input") == {str(_task(failing, k).partition) for k in (0, 1)}
    assert _by_stage(done1, "replay-output") == {str(_task(failing, 0).partition)}

    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a plan without store= opened a store")

    monkeypatch.setattr(gcp.Store, "__init__", refuse)
    monkeypatch.setattr(gcp.FsspecStore, "__init__", refuse)
    assert SequentialRunner().run(_plan(y)).value == 2.0 * sum(range(N))


def test_capture_through_a_url_from_a_spawned_process(tmp_path: Path) -> None:
    _s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    uri = (tmp_path / "cap").as_uri()
    plan = _plan(y, store=uri)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(2, mp_context=ctx) as ex:
        parts = list(ex.map(_run_task, [plan.process] * 4, [t.partition for t in plan.tasks]))
    assert sum(parts) == 2.0 * sum(range(N))
    done = gcp.FsspecStore(uri).completed()
    assert len(_by_stage(done, "replay-input")) == 4
    assert len(_by_stage(done, "replay-output")) == 4
    r = gd.replay(plan, 2, y)
    assert r.input_source == "store"
    d = r.diff()
    assert d.reference == "recorded"
    assert d.equal is True
    assert d.recorded == d.replayed == parts[[t.key for t in plan.tasks].index(2)]


def test_steps_walk_the_opt_level_0_cone_with_user_frames(tmp_path: Path) -> None:
    s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    gak.sum(ev.x + 1.0, axis=None)  # an unrelated program in the same arena
    plan = _plan(y)
    steps = list(gd.replay(plan, 0, y).steps())
    low = gd.lower(s, y, opt_level=0)
    assert [st.node for st in steps] == sorted(low.ops, key=lambda o: o.node_id)
    assert [o.op for o in low.ops] == ["events", "field", "mul", "ak.sum"]
    assert low.ops[-1].form == "float64"
    by_op = {st.node.op: st for st in steps}
    assert ak.array_equal(by_op["field"].value, np.arange(float(ROWS)))
    assert ak.array_equal(by_op["mul"].value, 2.0 * np.arange(float(ROWS)))
    assert float(by_op["ak.sum"].value) == 2.0 * sum(range(ROWS))
    assert all(st.node.provenance.lineno == s.provenance(Array(s, st.node.node_id)).lineno for st in steps)
    assert len(steps) > len(GraphStore.deserialize(plan.process.ir).nodes())


def test_steps_are_lazy_and_timed(tmp_path: Path) -> None:
    _s, ev = _events(tmp_path)
    y = gak.sum(ev.x.map(_slow, name="slow") * 2.0, axis=None)
    plan = _plan(y)
    SequentialRunner().run(plan)
    base = len(_SLOW)
    r = gd.replay(plan, 0, y)
    it = r.steps()
    first = next(it)
    assert len(_SLOW) == base
    steps = [first, *it]
    assert len(_SLOW) == base + 1
    (mapped,) = [st for st in steps if st.node.kind == "external"]
    after = steps[steps.index(mapped) + 1]
    assert mapped.seconds >= 0.05
    assert 0.0 <= after.seconds < mapped.seconds
    assert r.value == 2.0 * sum(range(ROWS))
    assert len(_SLOW) == base + 2
    assert r.value == 2.0 * sum(range(ROWS))
    assert len(_SLOW) == base + 2


def test_value_and_diff_against_the_recorded_output(tmp_path: Path) -> None:
    _s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    stored = _plan(y, store=tmp_path / "cap")
    SequentialRunner().run(stored)
    plain = _plan(y)
    expected = plain.process(_task(plain, 1).partition, LocalResources())
    assert expected == 2.0 * sum(range(ROWS, 2 * ROWS))

    r = gd.replay(stored, 1, y)
    assert r.input_source == "store"
    assert r.value == expected
    d = r.diff()
    assert (d.reference, d.equal, d.recorded, d.replayed) == ("recorded", True, expected, expected)

    r0 = gd.replay(plain, 1, y)
    assert r0.input_source == "re-read"
    assert r0.value == expected
    d0 = r0.diff()
    assert (d0.reference, d0.equal, d0.recorded, d0.replayed) == ("re-evaluated", True, expected, expected)


def test_inputs_come_from_the_store(tmp_path: Path) -> None:
    _s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    stored = _plan(y, store=tmp_path / "cap")
    plain = _plan(y)
    SequentialRunner().run(stored)
    expected = plain.process(_task(plain, 2).partition, LocalResources())
    os.remove(tmp_path / "d.parquet")

    assert gd.replay(stored, 2, y).value == expected
    r = gd.replay(plain, 2, y)
    # D-4/D-5 leave open whether a failed re-read surfaces raw or as the source node's StageError.
    with pytest.raises((FileNotFoundError, gd.StageError)) as ei:
        _ = r.value
    err = ei.value
    assert isinstance(err, FileNotFoundError) or isinstance(err.__cause__, FileNotFoundError)


def test_failing_task_raises_at_its_node(tmp_path: Path) -> None:
    s, ev = _events(tmp_path)
    m = ev.x.map(_fail_on_key1, name="fail")
    y = gak.sum(m * 2.0, axis=None)
    plan = _plan(y, store=tmp_path / "cap")
    with pytest.raises(gd.StageError):
        SequentialRunner().run(plan)
    part = str(_task(plan, 1).partition)

    r = gd.replay(plan, 1, y)
    assert r.input_source == "store"
    seen: list[str] = []
    with pytest.raises(gd.StageError) as ei:
        for st in r.steps():
            seen.append(st.node.op)
    assert seen == ["events", "field"]
    err = ei.value
    assert isinstance(err.__cause__, ValueError)
    want = _stage_error(gd.lower(s, Array(s, m.node_id), opt_level=0), m.node_id, part, err.__cause__)
    assert err.opt_level == 0
    assert err.op == "external" == want.op
    assert err.partition == part
    assert err.user_frame.lineno == s.provenance(m).lineno
    assert len(err.frames) >= 2
    assert (err.frames, err.input_forms) == (want.frames, want.input_forms)
    with pytest.raises(gd.StageError):
        r.diff()

    r2 = gd.replay(plan, 2, y)
    assert r2.input_source == "re-read"
    d2 = r2.diff()
    assert (d2.reference, d2.equal) == ("re-evaluated", True)
    assert d2.replayed == 2.0 * sum(range(2 * ROWS, 3 * ROWS))


def test_diff_reports_a_difference(tmp_path: Path) -> None:
    _DROP.clear()
    _s, ev = _events(tmp_path)
    rows = ev.x.map(_drop_after_first, name="drop")
    plan = _plan(rows, reduce=_rows, combine=_rows_add, empty=_rows_empty, store=tmp_path / "cap")
    SequentialRunner().run(plan)
    d = gd.replay(plan, 0, rows).diff()
    assert d.reference == "recorded"
    assert d.equal is False
    assert len(d.recorded["rows"][0]) == ROWS
    assert len(d.replayed["rows"][0]) == ROWS - 1
    assert ak.array_equal(d.recorded["rows"][0], np.arange(float(ROWS)))


def test_replay_refusals(tmp_path: Path) -> None:
    _s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    z = gak.sum(ev.x + 1.0, axis=None)
    plan = _plan(y)
    plain: Plan[float] = Plan(
        process=_plain, combine=_add, empty=_zero, tasks=[Task(0, Partition.blind("u", "t", 0, 1))]
    )
    with pytest.raises(TypeError, match="aggregate_plan"):
        gd.replay(plain, 0, y)
    with pytest.raises(ValueError, match="99"):
        gd.replay(plan, 99, y)
    with pytest.raises(ValueError, match="outputs"):
        gd.replay(plan, 0, z)


def test_replay_binds_the_runs_external_evaluators(tmp_path: Path) -> None:
    s, ev = _events(tmp_path)
    h, key = _external(s, ev.x)
    plan = _plan(h, externals={key: _ext_sum_plus_one})
    run = plan.process(_task(plan, 0).partition, LocalResources())
    assert run == float(sum(range(ROWS))) + 1.0
    assert gd.replay(plan, 0, h).value == run


def test_replay_reads_its_own_plans_capture_root(tmp_path: Path) -> None:
    """Plans over the same outputs and partitions (so the same capture ids), each captured into its
    own root as plan-D prescribes (one run per root), each diff against their own recorded output."""
    s, ev = _events(tmp_path)
    y = gak.sum(ev.x * 2.0, axis=None)
    a = _plan(y, store=tmp_path / "cap_a")
    b = _plan(y, reduce=_first_plus_one, store=tmp_path / "cap_b")
    SequentialRunner().run(a)
    SequentialRunner().run(b)
    for plan, want in ((a, 2.0 * sum(range(ROWS))), (b, 2.0 * sum(range(ROWS)) + 1.0)):
        d = gd.replay(plan, 0, y).diff()
        assert (d.reference, d.equal, d.recorded, d.replayed) == ("recorded", True, want, want)

    h, key = _external(s, ev.x)
    one = _plan(h, externals={key: _ext_sum_plus_one}, store=tmp_path / "cap_one")
    two = _plan(h, externals={key: _ext_sum_plus_two}, store=(tmp_path / "cap_two").as_uri())
    SequentialRunner().run(one)
    SequentialRunner().run(two)
    for plan, want in ((one, float(sum(range(ROWS))) + 1.0), (two, float(sum(range(ROWS))) + 2.0)):
        d = gd.replay(plan, 0, h).diff()
        assert (d.reference, d.equal, d.recorded, d.replayed) == ("recorded", True, want, want)
