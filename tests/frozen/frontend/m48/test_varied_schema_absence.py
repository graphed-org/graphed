"""§7.2: `Plan` / `ExecResult` / monitor-payload schemas do not change for a VARIED program.

Worded over KEY SETS against literally spelled sets, never against a sibling unvaried run — that
comparison is equal by construction even after a field is added. Every assertion here rides a
program that is genuinely varied: a version reading the schemas off an unvaried program would pass
identically before and after m48 and could never fail in the direction it guards.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from vary_fixtures import (
    add_per_output,
    loose_varied,
    partitioned_vector_source,
    sibling_outputs,
    sum_per_output,
)

import graphed
from graphed.core.execution import ExecResult, Plan, SequentialRunner, TaskEvent


class _CollectingMonitor:
    """A passive M37 monitor: the executor hands `on_task` the payload whose schema is at stake."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def on_task(self, event: Any) -> None:
        self.events.append(event)

    def on_profile(self, worker: str, payload: bytes) -> None:
        return None

    def on_combine(self, leaves_done: int) -> None:
        return None

    def worker_profiler_factory(self) -> None:
        return None


def _varied_run() -> tuple[Plan[list[float]], ExecResult[list[float]], _CollectingMonitor]:
    """A varied program lowered to siblings — one marked output per label — and executed."""
    _s, x, _src = partitioned_vector_source()
    varied = loose_varied(x)
    outputs = sibling_outputs(varied)
    assert len(outputs) == len(graphed.labels(varied)) == 3
    plan = graphed.aggregate_plan(
        *outputs,
        reduce=sum_per_output,
        combine=add_per_output,
        empty=lambda: [0.0, 0.0, 0.0],
        steps_per_file=3,
    )
    monitor = _CollectingMonitor()
    return plan, SequentialRunner(monitor).run(plan), monitor


def test_plan_schema_is_unchanged_by_a_varied_program() -> None:
    plan, _result, _monitor = _varied_run()
    assert {f.name for f in dataclasses.fields(plan)} == {
        "process",
        "combine",
        "empty",
        "tasks",
        "next_tasks",
        "stop",
        "open_once",
    }


def test_exec_result_schema_is_unchanged_by_a_varied_program() -> None:
    _plan, result, _monitor = _varied_run()
    assert {f.name for f in dataclasses.fields(result)} == {
        "value",
        "n_partitions",
        "n_combines",
        "stopped",
    }
    assert len(result.value) == 3  # the varied program really did reach a result


def test_monitor_task_payload_schema_is_unchanged_by_a_varied_program() -> None:
    _plan, _result, monitor = _varied_run()
    assert monitor.events, "no task was emitted, so the payload under test was never produced"
    event = monitor.events[0]
    assert isinstance(event, TaskEvent)  # `emit_task` ships the dataclass INSTANCE, not a dict
    assert {f.name for f in dataclasses.fields(event)} == {
        "phase",
        "key",
        "worker",
        "t",
        "partition",
        "n_entries",
        "bytes_read",
        "error",
    }
