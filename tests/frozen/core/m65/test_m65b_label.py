"""M65 B frozen suite (graphed-core slice): blind partitions label their step, and the
``SequentialRunner`` formats no label when no monitor is attached. No awkward."""

from __future__ import annotations

from typing import Any

import pytest

import graphed.core.execution as ce
from graphed.core import Partition, Plan, SequentialRunner, Task, TaskEvent, partition_label


def test_blind_partitions_label_their_step() -> None:
    labels = [partition_label(Partition.blind("/d/a.parquet", "", i, 4)) for i in range(4)]
    assert len(set(labels)) == 4
    assert labels[1] == "/d/a.parquet::1/4"
    assert partition_label(Partition("f.root", "Events", 0, 100)) == "f.root:Events:0-100"


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


def test_sequential_runner_formats_no_label_without_a_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Partition] = []
    real = ce.partition_label

    def counting(p: Partition) -> str:
        calls.append(p)
        return real(p)

    monkeypatch.setattr(ce, "partition_label", counting)
    tasks = [Task(k, Partition(f"f{k}.root", "Events", 0, k + 1)) for k in range(10)]
    plan: Plan[Any] = Plan(
        process=lambda p, r: p.n_entries, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks
    )

    bare = SequentialRunner().run(plan)
    assert calls == []
    monitored = SequentialRunner(Recorder()).run(plan)
    assert len(calls) > 0
    assert bare.value == monitored.value == sum(range(1, 11))
