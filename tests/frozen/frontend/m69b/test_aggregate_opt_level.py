"""`aggregate_plan(opt_level=)`: `1`, the default, is the optimized compile; `0` ships the 1:1 cone
of the plan's outputs, write arrays and metadata arrays; any other value is refused."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import m69b_opt_fixtures as fx
import pytest

from graphed import Array, aggregate_plan, compile_ir
from graphed.core import GraphStore
from graphed.core.execution import Plan, SequentialRunner
from graphed.debug import StageError, lower, replay
from graphed.write import PartWrite


def _plan(*outputs: Array, **kw: Any) -> Plan[Any]:
    return aggregate_plan(*outputs, reduce=fx.sums, combine=fx.add, empty=fx.zero, steps_per_file=2, **kw)


def _node_count(ir: bytes) -> int:
    return len(GraphStore.deserialize(bytes(ir)).nodes())


#: (the level a plan compiles at, the kwargs that ask for it)
_LEVELS: tuple[tuple[int, dict[str, Any]], ...] = ((1, {}), (0, {"opt_level": 0}))


def test_opt0_runs_and_reduce_receives_one_value_per_marked_output() -> None:
    _s, events = fx.record()
    outputs = fx.merged_pair(events)
    total = float(fx.COLUMNS["w"].sum())
    assert SequentialRunner().run(_plan(*outputs)).value.tolist() == [total]
    assert SequentialRunner().run(_plan(*outputs, opt_level=0)).value.tolist() == [total, total]


def test_opt0_ships_the_lowered_cone_of_the_outputs() -> None:
    session, events = fx.record()
    outputs = fx.merged_pair(events)
    cone = {op.node_id for o in outputs for op in lower(session, o, opt_level=0).ops}
    shipped = _node_count(_plan(*outputs, opt_level=0).process.ir)
    assert shipped == len(cone)
    assert _node_count(compile_ir(session, *outputs, optimize=False).ir) > shipped


def test_opt0_writes_the_defaults_parts(tmp_path: Path) -> None:
    arities: dict[int, list[int]] = {}
    parts: dict[int, dict[str, bytes]] = {}
    for level, kw in _LEVELS:
        _s, events = fx.record()
        dest = tmp_path / str(level)
        write = PartWrite(
            array=events.w + 1.0,
            destination=str(dest),
            name=fx.by_step,
            codec=fx.json_codec,
            metadata={"sumw": events.w.sum()},
        )
        plan = aggregate_plan(
            *fx.merged_pair(events),
            reduce=fx.arity,
            combine=fx.add,
            empty=fx.no_arities,
            steps_per_file=2,
            writes=[write],
            **kw,
        )
        arities[level] = SequentialRunner().run(plan).value
        parts[level] = {p.name: p.read_bytes() for p in dest.iterdir()}
    assert arities == {1: [2, 2], 0: [3, 3]}
    assert sorted(parts[0]) == ["part0.json", "part1.json"]
    assert parts[0] == parts[1]


def test_the_default_is_the_optimized_compile() -> None:
    session, events = fx.record()
    outputs = fx.merged_pair(events)
    implicit = _plan(*outputs).process.ir
    assert _plan(*outputs, opt_level=1).process.ir == implicit == bytes(compile_ir(session, *outputs).ir)


_CHILD = (
    "import sys; sys.path.insert(0, {here!r}); import m69b_opt_fixtures; m69b_opt_fixtures.emit_opt0_digest()"
)


def _child_digest(seed: str) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD.format(here=str(Path(__file__).resolve().parent))],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    lines = done.stdout.splitlines()
    assert len(lines) == 1, done.stdout
    return lines[0].split()


def test_opt0_ir_and_pickled_plan_do_not_depend_on_the_hash_seed() -> None:
    one, two = _child_digest("1"), _child_digest("2")
    assert one[2] != two[2], "the seeds did not change str hashing, so the comparison below is dead"
    assert one[:2] == two[:2]


@pytest.mark.parametrize(
    ("pair", "kw"),
    [
        pytest.param("merged", {"opt_level": 0}, id="w-and-w-times-1-at-0"),
        pytest.param("same", {"opt_level": 0}, id="w-twice-at-0"),
        pytest.param("merged", {}, id="w-and-w-times-1-by-default"),
    ],
)
def test_replay_of_a_capturing_plan_equals_its_run(tmp_path: Path, pair: str, kw: dict[str, int]) -> None:
    _s, events = fx.record()
    outputs = fx.merged_pair(events) if pair == "merged" else (events.w, events.w)
    plan = _plan(*outputs, store=str(tmp_path), **kw)
    SequentialRunner().run(plan)
    diff = replay(plan, 0, *outputs).diff()
    assert (diff.reference, diff.equal) == ("recorded", True)


def test_a_failing_op_reports_the_compiled_level_and_its_own_line() -> None:
    for level, kw in _LEVELS:
        _s, events = fx.record()
        failing, line = fx.failing_op(events)
        plan = _plan(failing, **kw)
        with pytest.raises(StageError) as info:
            SequentialRunner().run(plan)
        frame = info.value.user_frame
        assert (info.value.opt_level, Path(frame.filename).name, frame.lineno) == (
            level,
            "m69b_opt_fixtures.py",
            line,
        )


def test_cone_refuses_an_id_past_the_store() -> None:
    store = GraphStore()
    kept = store.add_op("neg", [store.add_source("events")])
    past = store.node_count()
    store.cone(outputs=[kept])
    with pytest.raises(ValueError, match=rf"no node with id {past}\b"):
        store.cone(outputs=[kept, past])


@pytest.mark.parametrize("level", [2, -1])
def test_other_levels_are_refused_naming_0_and_1(level: int) -> None:
    _s, events = fx.record()
    with pytest.raises(ValueError) as info:
        _plan(*fx.merged_pair(events), opt_level=level)
    assert {"0", "1"} <= set(re.findall(r"-?\d+", str(info.value)))
