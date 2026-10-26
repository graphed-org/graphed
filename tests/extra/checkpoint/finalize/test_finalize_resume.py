"""A checkpointed run finalizes its value once, whether it ran, resumed, or stopped."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graphed.checkpoint import Store, resumable, run_resumable
from graphed.checkpoint.resume import _Resumable
from graphed.checkpoint.runner import _SimulatedInterrupt
from graphed.core import DurablePlan, OpSpec, Partition, Plan, SequentialRunner

FINALIZED: list[int] = []


def process(p: Partition, _r: Any) -> int:
    if p.uri == "mem://bad":
        raise RuntimeError("bad partition")
    return p.entry_stop - p.entry_start


def combine(a: int, b: int) -> int:
    return a + b


def empty() -> int:
    return 0


def finalize(v: int) -> tuple[str, int]:
    FINALIZED.append(v)
    return ("final", v)


def _plan(*uris: str) -> DurablePlan:
    def op(fn: Any) -> OpSpec:
        return OpSpec("ref", f"t:{fn.__name__}", live=fn)

    return DurablePlan(
        ir=b"finalize-resume",
        process=op(process),
        combine=op(combine),
        empty=op(empty),
        partitions=tuple(Partition(u, "t", 0, i + 1) for i, u in enumerate(uris)),
        finalize=op(finalize),
    )


@pytest.fixture(autouse=True)
def _clear() -> None:
    FINALIZED.clear()


def test_a_resumed_run_finalizes_once(tmp_path: Path) -> None:
    plan, store = _plan("mem://a", "mem://b", "mem://c"), Store(tmp_path)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan, store, _kill_after=1)
    assert FINALIZED == []
    res = run_resumable(plan, store)
    assert res.value == ("final", 6) and FINALIZED == [6]
    assert res.report.skipped == 1


def test_an_empty_and_a_stopped_run_finalize_what_they_return(tmp_path: Path) -> None:
    assert run_resumable(_plan(), Store(tmp_path / "e")).value == ("final", 0)
    stopping = _plan("mem://a", "mem://bad", "mem://c")
    res = run_resumable(stopping, Store(tmp_path / "s"), error_budget=0)
    assert res.report.stopped == "error_budget"
    assert res.value == ("final", 1) and FINALIZED == [0, 1]


def test_resumable_carries_finalize_onto_the_runtime_plan(tmp_path: Path) -> None:
    plan = _plan("mem://a", "mem://b")
    for reused in (0, 2):
        runtime = resumable(plan, str(tmp_path))
        assert isinstance(runtime, Plan) and isinstance(runtime.process, _Resumable)
        assert runtime.process.reused == reused
        assert SequentialRunner().run(runtime).value == ("final", 3)
    assert FINALIZED == [3, 3]
