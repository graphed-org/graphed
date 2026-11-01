"""SequentialRunner has a pooled runner's surface: `with`, `submit`, `close`, `max_in_flight`."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import pytest

from graphed.core import ExecResult, Partition, Plan, SequentialRunner, Task


def _plan(*entries: int, fail: bool = False) -> Plan[int]:
    def process(partition: Partition, resources: Any) -> int:
        if fail:
            raise ValueError("bad chunk")
        return partition.n_entries

    tasks = tuple(Task(k, Partition("f.root", "Events", 0, n)) for k, n in enumerate(entries))
    return Plan(process=process, combine=lambda a, b: a + b, empty=lambda: 0, tasks=tasks)


def test_with_block_runs_a_plan() -> None:
    with SequentialRunner() as runner:
        assert isinstance(runner, SequentialRunner)
        assert runner.run(_plan(1, 2)).value == 3


def test_submit_returns_a_finished_future_of_the_result() -> None:
    fut = SequentialRunner().submit(_plan(1, 2))
    assert isinstance(fut, Future)
    assert fut.done()
    result = fut.result()
    assert isinstance(result, ExecResult)
    assert (result.value, result.n_partitions) == (3, 2)


def test_submit_carries_a_failure_in_the_future() -> None:
    fut = SequentialRunner().submit(_plan(1, fail=True))
    assert isinstance(fut.exception(), ValueError)
    with pytest.raises(ValueError, match="bad chunk"):
        fut.result()


def test_submitted_plans_run_in_submit_order() -> None:
    finished: list[int] = []
    with SequentialRunner() as runner:
        for n in (3, 1, 2):
            runner.submit(_plan(n)).add_done_callback(lambda f: finished.append(f.result().value))
    assert finished == [3, 1, 2]


def test_close_is_safe_twice_and_the_runner_still_runs() -> None:
    runner = SequentialRunner()
    runner.close()
    runner.close()
    assert runner.run(_plan(4)).value == 4
    assert runner.max_in_flight == 1


def test_exit_does_not_swallow_the_failure() -> None:
    with pytest.raises(ValueError, match="bad chunk"), SequentialRunner() as runner:
        runner.run(_plan(1, fail=True))
