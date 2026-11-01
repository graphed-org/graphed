"""A run goes on past ``skip_failed`` failed chunks and says which."""

from __future__ import annotations

import dataclasses
import json
import operator
import pickle
from collections.abc import Mapping
from typing import Any

import pytest

from graphed.core import (
    DurablePlanV2,
    ExecContext,
    FailedChunk,
    Partition,
    Plan,
    SequentialRunner,
    StopCondition,
    StopReason,
    Task,
    TooManyFailedChunksError,
    chunk_label,
    n_failed,
    skipping,
    without_lost,
)

N = 20


class Chunks:
    def __init__(self, *bad: int, flaky: int = 0) -> None:
        self.bad = set(bad)
        self.flaky = flaky

    def __call__(self, part: Partition, res: object) -> int:
        if self.flaky:
            self.flaky -= 1
            raise OSError("flaky")
        if part.entry_start in self.bad:
            raise OSError(f"bad chunk {part.entry_start}")
        return 1 << part.entry_start


def plan_of(process: Any, **kw: Any) -> Plan[int]:
    tasks = tuple(Task(i, Partition("f.root", "Events", i, i + 1)) for i in range(N))
    return Plan(process=process, combine=operator.add, empty=int, tasks=tasks, **kw)


def label(i: int) -> str:
    return chunk_label(Partition("f.root", "Events", i, i + 1))


def test_the_value_is_the_merge_of_the_other_chunks() -> None:
    with pytest.warns(UserWarning, match=r"1 failed chunk\(s\) are missing"):
        res = SequentialRunner().run(plan_of(Chunks(7)), skip_failed=10)
    assert res.value == (1 << N) - 1 - (1 << 7)
    assert [f.partition for f in res.failed] == [label(7)]
    assert res.failed[0].error == "OSError: bad chunk 7"
    assert res.failed[0].record["error_type"] == "OSError"


def test_a_clean_run_reports_nothing(recwarn: pytest.WarningsRecorder) -> None:
    res = SequentialRunner().run(plan_of(Chunks()), skip_failed=10)
    assert res.value == (1 << N) - 1
    assert res.failed == ()
    assert not recwarn


def test_one_failure_past_the_budget_raises_and_lists_them() -> None:
    with pytest.raises(TooManyFailedChunksError) as info:
        SequentialRunner().run(plan_of(Chunks(*range(11))), skip_failed=10)
    assert [f.partition for f in info.value.failed] == [label(i) for i in range(11)]
    assert str(info.value).startswith("11 chunks failed:\n  f.root:Events (entries 0-1): OSError")


def test_the_budget_allows_exactly_its_count() -> None:
    with pytest.warns(UserWarning, match="10 failed"):
        res = SequentialRunner().run(plan_of(Chunks(*range(10))), skip_failed=10)
    assert len(res.failed) == 10


def test_failures_read_in_label_order_whatever_the_merge_order() -> None:
    with pytest.warns(UserWarning, match="2 failed"):
        res = SequentialRunner().run(plan_of(Chunks(1, 2)), skip_failed=2)
    assert [f.partition for f in res.failed] == [label(1), label(2)]


def test_no_budget_raises_on_the_first_error() -> None:
    with pytest.raises(OSError, match="bad chunk 2"):
        SequentialRunner().run(plan_of(Chunks(2, 3)))


def test_max_errors_is_the_same_budget() -> None:
    plan = plan_of(Chunks(1, 2), stop=StopCondition(max_errors=5))
    with pytest.warns(UserWarning, match="2 failed"):
        assert len(SequentialRunner().run(plan).failed) == 2
    with pytest.raises(TooManyFailedChunksError):
        SequentialRunner().run(plan_of(Chunks(1, 2), stop=StopCondition(max_errors=1)))


def test_skipping_makes_a_budget_stop_the_stream() -> None:
    wrapped = skipping(plan_of(Chunks()), 4)
    assert isinstance(wrapped, Plan)
    assert wrapped.stop == StopCondition(max_errors=4)
    assert wrapped.stop.reason(ExecContext(errors=5)) is StopReason.ERROR_BUDGET


def test_failures_round_trip_through_json() -> None:
    with pytest.warns(UserWarning):
        res = SequentialRunner().run(plan_of(Chunks(3)), skip_failed=1)
    (failed,) = res.failed
    assert FailedChunk(**json.loads(json.dumps(dataclasses.asdict(failed)))) == failed


def test_the_budget_error_crosses_a_process_boundary() -> None:
    failed = (FailedChunk("a", "OSError: x", {}),)
    back = pickle.loads(pickle.dumps(TooManyFailedChunksError(failed)))
    assert back.failed == failed
    assert "a: OSError: x" in str(back)


def test_attempts_run_inside_the_worker() -> None:
    wrapped: Any = skipping(plan_of(Chunks(flaky=1)), 1, attempts=2)
    assert n_failed(wrapped.process(wrapped.tasks[0].partition, None)) == 0


def test_one_attempt_fails_a_flaky_chunk() -> None:
    wrapped: Any = skipping(plan_of(Chunks(flaky=1)), 1)
    assert n_failed(wrapped.process(wrapped.tasks[0].partition, None)) == 1


def test_an_interrupt_is_not_a_failed_chunk() -> None:
    def interrupted(part: Partition, res: object) -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SequentialRunner().run(plan_of(interrupted), skip_failed=1)


def test_n_failed_of_a_plain_partial_is_zero() -> None:
    assert n_failed(3) == 0


class LostError(Exception):
    def __init__(self, partition: str) -> None:
        super().__init__(partition)
        self.partition = partition


def _skipping(**kw: Any) -> Any:
    return skipping(plan_of(Chunks(), **kw), 3)


def test_a_lost_chunk_fails_on_the_rerun_and_keeps_the_tree() -> None:
    err = LostError(label(4))
    rerun = without_lost(_skipping(), err)
    runner = SequentialRunner()
    with pytest.warns(UserWarning, match="1 failed"):
        res = runner.run(rerun)
    assert res.value == (1 << N) - 1 - (1 << 4)
    assert res.n_partitions == N
    assert [f.error for f in res.failed] == [f"LostError: {label(4)}"]


def test_a_second_lost_chunk_adds_to_the_first() -> None:
    rerun = without_lost(without_lost(_skipping(), LostError(label(1))), LostError(label(2)))
    with pytest.warns(UserWarning, match="2 failed"):
        assert len(SequentialRunner().run(rerun).failed) == 2


@pytest.mark.parametrize(
    "plan",
    [plan_of(Chunks()), _skipping(next_tasks=lambda ctx: None)],
    ids=["not-skipping", "pull-stream"],
)
def test_a_plan_that_cannot_rerun_raises_the_loss(plan: Plan[Any]) -> None:
    err = LostError(label(1))
    with pytest.raises(LostError) as info:
        without_lost(plan, err)
    assert info.value is err


def test_a_loss_outside_any_chunk_raises() -> None:
    with pytest.raises(LostError):
        without_lost(_skipping(), LostError("no chunk (a combine, or idle)"))


def test_the_same_chunk_lost_twice_raises() -> None:
    once = without_lost(_skipping(), LostError(label(1)))
    with pytest.raises(LostError):
        without_lost(once, LostError(label(1)))


class Bound:
    def __init__(self, endpoints: Mapping[str, str] | None = None) -> None:
        self.endpoints = endpoints

    def __call__(self, part: Partition, res: object) -> int:
        return 1

    def bind_services(self, endpoints: Mapping[str, str]) -> Bound:
        return Bound(endpoints)


def test_services_bind_through_the_wrapper() -> None:
    wrapped: Any = skipping(plan_of(Bound()), 1)
    bound = wrapped.process.bind_services({"a": "tcp://h:1"})
    assert bound.process.endpoints == {"a": "tcp://h:1"}


def test_an_unbindable_process_binds_to_itself() -> None:
    wrapped: Any = skipping(plan_of(Chunks()), 1)
    assert wrapped.process.bind_services({}) is wrapped.process


def test_a_staged_plan_has_no_chunks_to_skip() -> None:
    staged = object.__new__(DurablePlanV2)
    assert skipping(staged) is staged
    with pytest.raises(TypeError, match="needs a Plan"):
        skipping(staged, 1)


def test_no_budget_leaves_the_plan_alone() -> None:
    plan = plan_of(Chunks())
    assert skipping(plan) is plan


def test_a_plan_is_wrapped_once() -> None:
    wrapped = skipping(plan_of(Chunks()), 2)
    assert skipping(wrapped) is wrapped
