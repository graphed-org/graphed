"""m72 (b): graphs over different sources collate into one executable plan."""

from __future__ import annotations

import multiprocessing
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from m72_multiout_fixtures import (
    DATA_FILES,
    MC_FILES,
    concat,
    data_plan,
    echo_uri,
    mc_plan,
    nothing,
    part_bytes,
    read_part,
    tree_fold,
)

import graphed
import graphed.aggregate
from graphed.core import Partition
from graphed.core.execution import LocalResources, Plan, SequentialRunner, StopCondition, Task


def _hand(*tasks: Task, **kwargs: Any) -> Plan[list[str]]:
    return Plan(process=echo_uri, combine=concat, empty=nothing, tasks=tasks, **kwargs)


def _key_ordered(plan: Plan[Any]) -> list[Task]:
    return sorted(plan.tasks, key=lambda t: t.key)


def _fold(plan: Plan[Any]) -> Any:
    partials = [plan.process(t.partition, LocalResources()) for t in _key_ordered(plan)]
    return tree_fold(plan.combine, partials)


def test_collate_value_equals_each_plan_run_alone() -> None:
    assert graphed.collate is graphed.aggregate.collate
    mc, _ = mc_plan(2)
    data, _ = data_plan(3)
    collated = graphed.collate({"mc": mc, "data": data})
    value = SequentialRunner().run(collated).value
    alone = {"mc": SequentialRunner().run(mc_plan(2)[0]).value, "data": SequentialRunner().run(data_plan(3)[0]).value}
    assert value == alone
    assert value["mc"][1] and len(value["mc"][1]) == len(mc.tasks)
    assert _fold(collated) == value

    empty = _hand()
    with_empty = graphed.collate({"mc": mc_plan(2)[0], "none": empty})
    value = SequentialRunner().run(with_empty).value
    assert value == {"mc": alone["mc"]}
    assert _fold(with_empty) == value
    assert SequentialRunner().run(graphed.collate({"a": _hand(), "b": _hand()})).value == {}


def test_collate_routes_each_task_to_its_own_graph() -> None:
    mc, mc_src = mc_plan(2)
    data, data_src = data_plan(2)
    SequentialRunner().run(graphed.collate({"mc": mc, "data": data}))
    assert len(mc_src.reads) == len(mc.tasks) == 4
    assert len(data_src.reads) == len(data.tasks) == 4
    assert {(u, t) for u, t, *_ in mc_src.reads} == {(u, "Events") for u in MC_FILES}
    assert {(u, t) for u, t, *_ in data_src.reads} == {(u, "Runs") for u in DATA_FILES}

    first = _hand(Task(5, Partition("u1", "", 0, 1)), Task(2, Partition("u1", "", 1, 2)))
    second = _hand(Task(0, Partition("u2", "", 0, 1)))
    collated = graphed.collate({"z": first, "a": second})
    ordered = _key_ordered(collated)
    assert [t.key for t in ordered] == [0, 1, 2]
    assert [t.partition for t in ordered] == [
        Partition("u1", "", 1, 2), Partition("u1", "", 0, 1), Partition("u2", "", 0, 1)
    ]
    assert SequentialRunner().run(collated).value == {"z": ["u1:1", "u1:0"], "a": ["u2:0"]}


def test_collate_ships_no_per_task_data() -> None:
    def build(steps: int) -> Plan[Any]:
        return graphed.collate({"mc": mc_plan(steps)[0], "data": data_plan(steps)[0]})

    one, many = build(1), build(16)
    assert len(many.tasks) == 16 * len(one.tasks)
    assert len(pickle.dumps(one.process)) == len(pickle.dumps(many.process))
    subs = [mc_plan(16)[0], data_plan(16)[0]]
    expected = [t.partition for sub in subs for t in _key_ordered(sub)]
    ordered = _key_ordered(many)
    assert [t.partition for t in ordered] == expected
    for t in ordered:
        assert type(t) is Task
        assert pickle.dumps(t) == pickle.dumps(Task(t.key, t.partition))


def test_collate_refusals() -> None:
    task = Task(0, Partition("u", "t", 0, 1))
    with pytest.raises(ValueError, match="at least one plan"):
        graphed.collate({})
    with pytest.raises(TypeError, match="fixed tasks"):
        graphed.collate({"a": _hand(task, next_tasks=lambda ctx: None)})
    with pytest.raises(TypeError, match="fixed tasks"):
        graphed.collate({"a": _hand(task, stop=StopCondition(target_events=1))})
    with pytest.raises(ValueError, match="appears in plans") as err:
        graphed.collate({"left": _hand(task), "right": _hand(Task(0, Partition("u", "t", 1, 2)))})
    assert "left" in str(err.value) and "right" in str(err.value)
    graphed.collate({"left": _hand(task), "right": _hand(Task(0, Partition("u", "other", 0, 1)))})

    other = Task(0, Partition("v", "t", 0, 1))
    assert graphed.collate({"a": _hand(task), "b": _hand(other, open_once=True)}).open_once is True
    assert graphed.collate({"a": _hand(task, open_once=True), "b": _hand(other)}).open_once is True
    assert graphed.collate({"a": _hand(task), "b": _hand(other)}).open_once is False


def test_collated_plan_with_writes_runs_across_processes(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    mc, _ = mc_plan(2, str(parts / "mc"))
    data, _ = data_plan(2, str(parts / "data"))
    plan = graphed.collate({"mc": mc, "data": data})
    pickle.loads(pickle.dumps(plan.process))
    pickle.loads(pickle.dumps(list(plan.tasks)))

    sequential = SequentialRunner().run(plan).value
    written = part_bytes(str(parts))
    assert len(written) == len(plan.tasks)
    assert sorted(sequential["mc"][2] + sequential["data"][1]) == sorted(str(parts / p) for p in written)
    assert all(read_part(p)["kv"] == {"kind": "Data"} for p in sequential["data"][1])
    shutil.rmtree(parts)

    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {t.key: pool.submit(plan.process, t.partition, LocalResources()) for t in plan.tasks}
        partials = [futures[k].result() for k in sorted(futures)]
    assert tree_fold(plan.combine, partials) == sequential
    assert part_bytes(str(parts)) == written
