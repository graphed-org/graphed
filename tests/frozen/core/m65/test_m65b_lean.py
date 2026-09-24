"""M65 B frozen suite (graphed-core slice): lean events and the per-worker push hook on the monitor
seam, read through ``graphed.core`` helpers. Plain-Python process functions only (no awkward)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import graphed.core as gc
from graphed.core import Partition, Plan, Task, TaskEvent, TaskPhase, partition_label


class Recorder:
    def __init__(self) -> None:
        self.events: list[TaskEvent] = []

    def on_task(self, event: TaskEvent) -> None:
        self.events.append(event)

    def on_profile(self, worker: str, payload: bytes) -> None:
        return None

    def on_combine(self, leaves_done: int) -> None:
        return None

    def worker_profiler_factory(self) -> None:
        return None

    def phases(self, key: int) -> list[TaskPhase]:
        return [e.phase for e in self.events if e.key == key]


class LeanRecorder(Recorder):
    lean_events = True


class TruthyRecorder(Recorder):
    lean_events = 1


def _plan(n: int, fail_key: int | None = None) -> Plan[int]:
    def process(p: Partition, _reader: object) -> int:
        if p.entry_start == fail_key:
            raise ValueError(f"m65b boom {fail_key}")
        return p.n_entries

    tasks = [Task(k, Partition(f"f{k}.root", "Events", k, 2 * k + 3)) for k in range(n)]
    return Plan(process=process, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)


def test_sequential_lean_emits_submitted_and_terminal_only() -> None:
    plan = _plan(4)
    rec = LeanRecorder()
    res = gc.SequentialRunner(monitor=rec).run(plan)
    bare = gc.SequentialRunner().run(plan)
    assert (res.value, res.n_partitions, res.n_combines) == (bare.value, bare.n_partitions, bare.n_combines)
    labels = {t.key: partition_label(t.partition) for t in plan.tasks}
    for k in range(4):
        assert rec.phases(k) == [TaskPhase.SUBMITTED, TaskPhase.FINISHED]
    for e in rec.events:
        if e.phase is TaskPhase.SUBMITTED:
            assert e.partition == labels[e.key]
        else:
            assert e.partition == ""

    failing = _plan(4, fail_key=2)
    rec = LeanRecorder()
    with pytest.raises(ValueError, match="m65b boom 2"):
        gc.SequentialRunner(monitor=rec).run(failing)
    assert rec.phases(0) == rec.phases(1) == [TaskPhase.SUBMITTED, TaskPhase.FINISHED]
    assert rec.phases(2) == [TaskPhase.SUBMITTED, TaskPhase.ERRORED]
    (err,) = [e for e in rec.events if e.phase is TaskPhase.ERRORED]
    assert "m65b boom 2" in (err.error or "")
    assert err.partition == ""


def test_lean_opt_in_is_exactly_true() -> None:
    assert gc.lean_events(None) is False
    assert gc.lean_events(Recorder()) is False
    assert gc.lean_events(TruthyRecorder()) is False
    assert gc.lean_events(LeanRecorder()) is True

    full = TruthyRecorder()
    gc.SequentialRunner(monitor=full).run(_plan(3))
    for k in range(3):
        assert full.phases(k) == [TaskPhase.SUBMITTED, TaskPhase.STARTED, TaskPhase.FINISHED]
    assert all(e.partition for e in full.events)


def _factory() -> Recorder:
    return Recorder()


class Pushing(Recorder):
    def worker_monitor_factory(self) -> Callable[[], Any]:
        return _factory


def test_worker_monitor_factory_helper() -> None:
    assert gc.worker_monitor_factory(None) is None
    assert gc.worker_monitor_factory(Recorder()) is None
    assert gc.worker_monitor_factory(Pushing()) is _factory
