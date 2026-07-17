"""M37 frozen suite (graphed-core slice): the passive monitor seam in ``graphed.core.execution``.

The contract: an executor *emits* ``TaskEvent``s through a ``Monitor``; the seam is data-only and
purely passive — attaching a monitor MUST NOT change the reduced result, and a monitor that raises
MUST NOT break the run. ``SequentialRunner`` is the observable baseline.
"""

from __future__ import annotations

import dataclasses

import pytest

from graphed.core import (
    Monitor,
    Partition,
    Plan,
    SequentialRunner,
    Task,
    TaskEvent,
    TaskPhase,
    WorkerProfiler,
    emit_task,
    partition_label,
)


class Recorder:
    """A minimal in-process Monitor that records the event stream."""

    def __init__(self) -> None:
        self.events: list[TaskEvent] = []
        self.profiles: list[tuple[str, bytes]] = []
        self.combines = 0

    def on_task(self, event: TaskEvent) -> None:
        self.events.append(event)

    def on_profile(self, worker: str, payload: bytes) -> None:
        self.profiles.append((worker, payload))

    def on_combine(self, leaves_done: int) -> None:
        self.combines += 1

    def worker_profiler_factory(self) -> None:
        return None


def _plan(n: int = 5) -> Plan[int]:
    tasks = [Task(k, Partition(f"f{k}.root", "Events", 0, (k + 1) * 10)) for k in range(n)]
    return Plan(process=lambda p, r: p.n_entries, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)


def test_taskevent_is_frozen_with_expected_fields() -> None:
    ev = TaskEvent(TaskPhase.STARTED, key=3, worker="w0", t=1.0, partition="f:Events:0-10", n_entries=10)
    assert ev.phase is TaskPhase.STARTED and ev.key == 3 and ev.n_entries == 10
    assert ev.bytes_read is None and ev.error is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.key = 4  # type: ignore[misc]


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(Recorder(), Monitor)

    class Prof:
        def start(self) -> None: ...
        def flush(self) -> bytes | None:
            return None

        def stop(self) -> bytes | None:
            return None

    assert isinstance(Prof(), WorkerProfiler)


def test_partition_label_is_human_readable() -> None:
    assert partition_label(Partition("f.root", "Events", 0, 100)) == "f.root:Events:0-100"


def test_sequential_runner_emits_full_phase_sequence_in_key_order() -> None:
    plan = _plan(4)
    rec = Recorder()
    SequentialRunner(monitor=rec).run(plan)
    submitted = [e.key for e in rec.events if e.phase is TaskPhase.SUBMITTED]
    started = [e.key for e in rec.events if e.phase is TaskPhase.STARTED]
    finished = [e.key for e in rec.events if e.phase is TaskPhase.FINISHED]
    assert submitted == [0, 1, 2, 3]  # one SUBMITTED per task, in key order
    assert started == [0, 1, 2, 3]
    assert finished == [0, 1, 2, 3]
    # every task: exactly one of each phase
    for k in range(4):
        phases = [e.phase for e in rec.events if e.key == k]
        assert phases == [TaskPhase.SUBMITTED, TaskPhase.STARTED, TaskPhase.FINISHED]


def test_monitor_is_passive_result_identical() -> None:
    plan = _plan(6)
    bare = SequentialRunner().run(plan)
    observed = SequentialRunner(monitor=Recorder()).run(plan)
    assert observed.value == bare.value
    assert observed.n_partitions == bare.n_partitions
    assert observed.n_combines == bare.n_combines


def test_errored_task_emits_errored_then_propagates() -> None:
    def boom(p: Partition, r: object) -> int:
        raise ValueError("kaboom")

    tasks = [Task(0, Partition("f.root", "Events", 0, 10))]
    plan = Plan(process=boom, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)
    rec = Recorder()
    with pytest.raises(ValueError, match="kaboom"):  # the run still raises intact
        SequentialRunner(monitor=rec).run(plan)
    errored = [e for e in rec.events if e.phase is TaskPhase.ERRORED]
    assert len(errored) == 1
    assert "kaboom" in (errored[0].error or "")


def test_emit_task_swallows_a_raising_monitor() -> None:
    class Bad:
        def on_task(self, event: TaskEvent) -> None:
            raise RuntimeError("monitor exploded")

        def on_profile(self, worker: str, payload: bytes) -> None: ...
        def on_combine(self, leaves_done: int) -> None: ...
        def worker_profiler_factory(self) -> None:
            return None

    emit_task(Bad(), TaskEvent(TaskPhase.STARTED, 0, "w", 0.0))  # must not raise
    emit_task(None, TaskEvent(TaskPhase.STARTED, 0, "w", 0.0))  # None monitor is a no-op

    plan = _plan(3)
    result = SequentialRunner(monitor=Bad()).run(plan)  # a raising monitor cannot break the run
    assert result.value == sum((k + 1) * 10 for k in range(3))
