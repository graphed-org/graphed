"""m74 T1, T5-T10: resuming a plan on SequentialRunner and through direct ``process`` calls (plan §2.2,
§2.4, §2.6, §2.9, §4.1)."""

from __future__ import annotations

import functools
import gc
import pickle
import shutil
import struct
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import m74_helpers as h
import pytest

from graphed import Session, repartition, shuffle_plan
from graphed.awkward import AwkwardBackend, gak
from graphed.awkward.io import from_parquet
from graphed.checkpoint import Store, StoreUnavailable, resumable, run_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt
from graphed.core import DurablePlanV2, LocalResources, Partition, Plan, SequentialRunner
from graphed.services import bind_services, resolve_services


def test_resume_runs_only_the_tasks_not_done(tmp_path: Path) -> None:
    # T1
    k = 3
    marker = tmp_path / "calls.txt"
    plan = h.plan_of(h.Marker(str(marker)))
    root = str(tmp_path / "store")
    h.reset(interrupt_at=k + 1)
    with pytest.raises(h.Interrupt):
        SequentialRunner().run(resumable(plan, root))
    assert len(h.marks(marker)) == k
    assert len(Store(root).completed()) == k
    marker.unlink()
    h.reset()
    rp = resumable(plan, root)
    assert rp.process.reused == k
    value = SequentialRunner().run(rp).value
    assert sorted(h.marks(marker)) == sorted(str(p.entry_start) for p in h.partitions()[k:])
    assert value == SequentialRunner().run(h.plan_of(h.Marker(str(tmp_path / "reference.txt")))).value


def test_inner_process_sees_the_original_partitions(tmp_path: Path) -> None:
    # T5
    parts = (*h.partitions(3), Partition.blind("mem://m74/blind", "Events", 1, 4))
    rp = resumable(h.plan_of(h.Recorder(), parts), str(tmp_path / "store"))
    assert all(type(t.partition) is not Partition and t.partition.task_id for t in rp.tasks)
    h.RECEIVED.clear()
    SequentialRunner().run(rp)
    assert [(kind, p) for kind, p, _ in h.RECEIVED] == [(Partition, p) for p in parts]


def test_bind_and_resolve_services_reach_the_inner_process(tmp_path: Path) -> None:
    # T5
    rp = resumable(h.plan_of(h.Recorder()), str(tmp_path / "store"))
    assert not isinstance(rp.process, h.Recorder)
    bound = bind_services(rp, {"svc": "tcp://host:1"})
    assert type(bound.process) is type(rp.process)
    assert bound.process is not rp.process
    h.RECEIVED.clear()
    value = SequentialRunner().run(bound).value
    assert {endpoint for _, _, endpoint in h.RECEIVED} == {"tcp://host:1"}
    assert resolve_services(bound, value) == ("resolved", value)


@pytest.mark.parametrize("damage", ["deleted", "corrupted"])
def test_a_blob_lost_after_resumable_returns_is_recomputed(tmp_path: Path, damage: str) -> None:
    # T5
    root, marker = tmp_path / "store", tmp_path / "calls.txt"
    plan = h.plan_of(h.Marker(str(marker)))
    expected = SequentialRunner().run(resumable(plan, str(root))).value
    marker.unlink()
    rp = resumable(plan, str(root))
    victim = rp.tasks[2].partition
    assert victim.blob
    blob = root / "objects" / victim.blob
    if damage == "deleted":
        blob.unlink()
    else:
        blob.write_bytes(b"not the stored partial")
    assert SequentialRunner().run(rp).value == expected
    assert h.marks(marker) == [str(victim.entry_start)]


@pytest.mark.parametrize("kind", ["dir", "memory", "file"])
def test_three_live_threads_write_three_journals(tmp_path: Path, kind: str) -> None:
    # T6
    root = {
        "dir": str(tmp_path / "store"),
        "memory": f"memory://m74-{uuid.uuid4().hex}",
        "file": "file://" + (tmp_path / "store").as_posix(),
    }[kind]
    rp = resumable(h.plan_of(h.Barriered(), h.partitions(3)), root)
    h.BARRIER[:] = [threading.Barrier(3, timeout=10)]
    resources = LocalResources()
    with ThreadPoolExecutor(3) as pool:
        values = list(pool.map(lambda t: rp.process(t.partition, resources), rp.tasks))
    assert values == [0.0, 10.0, 20.0]
    journals = h.journal_records(root)
    writers = [name for name, records in journals.items() if any(r.get("stage") != "environment" for r in records)]
    assert len(writers) == 3


def test_durable_plan_becomes_a_runtime_plan_keyed_by_task_id(tmp_path: Path) -> None:
    # T7
    dp = h.durable("m74_helpers:marked")
    root = str(tmp_path / "store")
    h.reset()
    rp = resumable(dp, root)
    assert isinstance(rp, Plan)
    assert [t.key for t in rp.tasks] == list(range(h.T))
    assert [h.plain(t.partition) for t in rp.tasks] == list(dp.partitions)
    assert rp.process.reused == 0
    value = SequentialRunner().run(rp).value
    assert h.STATE["calls"] == h.T
    assert set(Store(root).completed()) == {t.partition.task_id for t in rp.tasks}
    assert value == run_resumable(dp, Store(tmp_path / "reference")).value
    h.reset()
    again = resumable(dp, root)
    assert again.process.reused == h.T
    assert SequentialRunner().run(again).value == value
    assert h.STATE["calls"] == 0


def _stage_payloads(plan: DurablePlanV2) -> list[list[bytes]]:
    resources, out = LocalResources(), []
    for stage in plan.stages:
        process = stage.process.resolve()
        upstream = tuple(p for dep in stage.inputs for p in out[dep])
        out.append([process(task, upstream, resources) for task in stage.tasks])
    return out


def test_shuffle_plan_interrupted_resumes_with_identical_gather_payloads(tmp_path: Path) -> None:
    # T7
    pytest.importorskip("pyarrow")
    ev = from_parquet(Session(AwkwardBackend()), "events", h.events(tmp_path), open_files=False, steps_per_file=2)
    plan = shuffle_plan(
        gak.sum(repartition(ev, n=2).x), reduce=h.StopReduce(), combine=h.agg_combine, empty=h.agg_empty, steps_per_file=2
    )
    root = str(tmp_path / "store")
    n_map = len(plan.stages[0].tasks)
    h.reset(interrupt_at=2)
    with pytest.raises(h.Interrupt):
        SequentialRunner().run(resumable(plan, root))
    assert len(Store(root).completed()) == n_map + 1
    h.reset()
    rp = resumable(plan, root)
    assert h.stage_reused(rp) == [n_map, 1, 0]
    resumed = SequentialRunner().run(rp).value
    h.reset()
    payloads = _stage_payloads(plan)
    assert resumed == plan.value(payloads[-1])
    final = resumable(plan, root)
    assert h.stage_reused(final) == [len(stage.tasks) for stage in plan.stages]
    store = Store(root)
    assert [store.get(t.partition.blob) for t in final.stages[1].tasks] == payloads[1]


def test_store_unavailable_wraps_store_calls_only(tmp_path: Path) -> None:
    # T8
    root = tmp_path / "store"
    rp = resumable(h.plan_of(h.marked), str(root))
    assert root.is_dir()
    shutil.rmtree(root)
    root.write_text("not a directory", encoding="utf-8")
    caught: list[BaseException] = []

    def call() -> None:
        try:
            rp.process(rp.tasks[0].partition, LocalResources())
        except BaseException as exc:
            caught.append(exc)

    worker = threading.Thread(target=call)
    worker.start()
    worker.join(60)
    assert len(caught) == 1
    err = caught[0]
    assert isinstance(err, StoreUnavailable)
    assert isinstance(err, RuntimeError)
    assert str(root) in str(err)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is StoreUnavailable
    assert str(back) == str(err)

    failing = resumable(h.plan_of(h.raising), str(tmp_path / "healthy"))
    with pytest.raises(ValueError, match="bad partition") as info:
        SequentialRunner().run(failing)
    assert type(info.value) is ValueError


def test_resumable_decodes_nothing_and_workers_decode(tmp_path: Path) -> None:
    # T9
    root = str(tmp_path / "store")
    plan = h.plan_of(h.marked)
    SequentialRunner().run(resumable(plan, root, codec=h.CountingCodec()))
    h.DECODES.clear()
    h.reset()
    rp = resumable(plan, root, codec=h.CountingCodec())
    assert (len(h.DECODES), h.STATE["calls"]) == (0, 0)
    assert rp.process.reused == h.T
    resources = LocalResources()

    def call(task: Any) -> int:
        rp.process(task.partition, resources)
        return threading.get_ident()

    with ThreadPoolExecutor(2) as pool:
        workers = set(pool.map(call, rp.tasks))
    assert len(h.DECODES) == h.T
    assert {thread for _, thread in h.DECODES} <= workers
    assert threading.get_ident() not in {thread for _, thread in h.DECODES}
    assert h.STATE["calls"] == 0


def test_run_resumable_folds_partials_as_they_come(tmp_path: Path) -> None:
    # T10
    order = h.durable("m74_helpers:leaf_tuple", "m74_helpers:concat", "m74_helpers:no_tuple", n=8)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(order, Store(tmp_path / "order"), _kill_after=3)
    res = run_resumable(order, Store(tmp_path / "order"))
    assert res.report.skipped == 3
    assert res.value == tuple(range(8))

    floats = h.durable("m74_helpers:leaf_float", n=8)
    expected = functools.reduce(h.add, [h.leaf_float(p, None) for p in floats.partitions])
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(floats, Store(tmp_path / "floats"), _kill_after=3)
    resumed = run_resumable(floats, Store(tmp_path / "floats")).value
    fresh = run_resumable(floats, Store(tmp_path / "fresh")).value
    assert struct.pack("<d", resumed) == struct.pack("<d", fresh) == struct.pack("<d", expected)

    partials = h.durable("m74_helpers:partial_of", "m74_helpers:add_partials", "m74_helpers:no_partial", n=8)
    for kill in (None, 3):
        where = tmp_path / f"partials-{kill}"
        if kill is not None:
            with pytest.raises(_SimulatedInterrupt) as info:
                run_resumable(partials, Store(where), _kill_after=kill)
            del info
        gc.collect()
        base = h.LIVE["now"]
        h.LIVE["peak"] = base
        value = run_resumable(partials, Store(where)).value
        assert h.LIVE["peak"] - base <= 3
        assert value.v == sum(10.0 * i for i in range(8))
        del value
