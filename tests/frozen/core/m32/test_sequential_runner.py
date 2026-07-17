"""M32 — the dependency-free reference executor lives with the execution contract.

`SequentialRunner` (and the `LocalResources` it uses) belong in `graphed_core.execution`, beside
the `Plan`/`Executor`/`WorkerResources` contract they implement — not in a frontend
write-format module, where it had accreted only because that was the first dependency-free
caller. Every layer that runs a plan (frontend writers, histogram aggregation, preservation,
benchmarks) now reaches it here without importing the executor package or anything write-shaped.
"""

from __future__ import annotations

from graphed_core import LocalResources, SequentialRunner
from graphed_core.execution import ExecResult, Executor, Partition, Plan, Task


def _count(partition: Partition, resources: object) -> list[int]:
    return [partition.entry_stop - partition.entry_start]


def _concat(a: list[int], b: list[int]) -> list[int]:
    return [*a, *b]


def _empty() -> list[int]:
    return []


def test_sequential_runner_satisfies_the_executor_protocol() -> None:
    assert isinstance(SequentialRunner(), Executor)


def test_runs_a_plan_to_the_reduced_result_in_key_order() -> None:
    tasks = tuple(Task(key=i, partition=Partition("d", "", i * 10, i * 10 + i)) for i in range(5))
    plan = Plan(process=_count, combine=_concat, empty=_empty, tasks=tasks)
    result = SequentialRunner().run(plan)
    assert isinstance(result, ExecResult)
    assert result.value == [0, 1, 2, 3, 4]  # widths, accumulated in ascending key order
    assert result.n_partitions == 5
    assert result.n_combines == 4


def test_empty_plan_yields_the_identity() -> None:
    result = SequentialRunner().run(Plan(process=_count, combine=_concat, empty=_empty, tasks=()))
    assert result.value == []
    assert result.n_partitions == 0 and result.n_combines == 0


def test_key_order_is_independent_of_task_order() -> None:
    parts = [Partition("d", "", i, i + 1) for i in range(4)]
    forward = SequentialRunner().run(
        Plan(
            process=lambda p, r: [p.entry_start],
            combine=_concat,
            empty=_empty,
            tasks=tuple(Task(i, parts[i]) for i in range(4)),
        )
    )
    shuffled = SequentialRunner().run(
        Plan(
            process=lambda p, r: [p.entry_start],
            combine=_concat,
            empty=_empty,
            tasks=tuple(Task(i, parts[i]) for i in (2, 0, 3, 1)),
        )
    )
    assert forward.value == shuffled.value == [0, 1, 2, 3]  # deterministic by key, not submit order


def test_local_resources_opens_each_uri_once() -> None:
    opens: list[str] = []

    def opener(uri: str) -> str:
        opens.append(uri)
        return uri.upper()

    res = LocalResources()
    assert res.open_once("a", opener) == "A"
    assert res.open_once("a", opener) == "A"  # cached
    assert res.open_once("b", opener) == "B"
    assert opens == ["a", "b"]  # 'a' opened exactly once


def test_resources_reach_the_process_via_open_once() -> None:
    seen: list[str] = []

    def proc(partition: Partition, resources: object) -> list[int]:
        resources.open_once(partition.uri, lambda u: seen.append(u) or u)  # type: ignore[attr-defined]
        return [1]

    tasks = tuple(Task(i, Partition("file", "", i, i + 1)) for i in range(3))
    SequentialRunner().run(Plan(process=proc, combine=_concat, empty=_empty, tasks=tasks))
    assert seen == ["file"]  # one runner => one LocalResources => 'file' opened once across tasks
