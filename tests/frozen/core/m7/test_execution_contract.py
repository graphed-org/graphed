"""The M7 execution contract (graphed-core owns it; reference executors are graphed-exec-local)."""

from __future__ import annotations

import pickle

import graphed_core as gc


def test_partition_entry_count() -> None:
    assert gc.Partition("f.root", "Events", 0, 1000).n_entries == 1000
    assert gc.Partition("f.root").n_entries == 0
    assert gc.Partition("f.root", entry_start=10, entry_stop=5).n_entries == 0  # never negative


def test_partition_and_task_are_picklable() -> None:
    t = gc.Task(key=3, partition=gc.Partition("f.root", "Events", 0, 100))
    assert pickle.loads(pickle.dumps(t)) == t  # tasks cross a process boundary


def test_stop_condition_target_events() -> None:
    stop = gc.StopCondition(target_events=1000)
    assert stop.reason(gc.ExecContext(events_done=999)) is None
    assert stop.reason(gc.ExecContext(events_done=1000)) is gc.StopReason.TARGET_EVENTS


def test_stop_condition_wall_clock_and_error_budget() -> None:
    assert gc.StopCondition(max_wall_s=1.0).reason(gc.ExecContext(elapsed_s=1.5)) is gc.StopReason.WALL_CLOCK
    assert gc.StopCondition(max_errors=2).reason(gc.ExecContext(errors=3)) is gc.StopReason.ERROR_BUDGET
    assert gc.StopCondition(max_errors=2).reason(gc.ExecContext(errors=2)) is None  # budget not exceeded


def test_no_stop_condition_means_run_to_exhaustion() -> None:
    assert gc.StopCondition().reason(gc.ExecContext(events_done=10**9)) is None


def test_plan_construction() -> None:
    plan: gc.Plan[int] = gc.Plan(
        process=lambda part, res: part.n_entries,
        combine=lambda a, b: a + b,
        empty=lambda: 0,
        tasks=[gc.Task(0, gc.Partition("f", "E", 0, 5))],
    )
    assert plan.tasks[0].partition.n_entries == 5
    assert plan.combine(2, 3) == 5
    assert plan.empty() == 0


def test_exec_result_fields() -> None:
    r: gc.ExecResult[int] = gc.ExecResult(value=42, n_partitions=4, n_combines=3)
    assert r.value == 42 and r.n_partitions == 4 and r.n_combines == 3
    assert r.stopped is None


def test_executor_is_runtime_checkable() -> None:
    class _Toy:
        def run(self, plan: gc.Plan[int]) -> gc.ExecResult[int]:
            return gc.ExecResult(plan.empty(), 0, 0)

    assert isinstance(_Toy(), gc.Executor)
    assert not isinstance(object(), gc.Executor)
