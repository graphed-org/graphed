"""``Plan.finalize``: once, on the head, after the last combine, on every value a run returns."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any

import cloudpickle

from graphed.core import (
    DurablePlan,
    DurablePlanV2,
    ExecResult,
    OpSpec,
    Partition,
    Plan,
    RunControl,
    SequentialRunner,
    StageSpec,
    StopReason,
    Task,
    finalize_result,
)

TASKS = tuple(Task(k, Partition("mem://x", "t", k, k + 1)) for k in range(4))


def _logged_plan(log: list[Any], tasks: Sequence[Task] = TASKS) -> Plan[Any]:
    def process(p: Partition, _r: Any) -> int:
        log.append(("process", p.entry_start))
        return p.entry_start + 1

    def combine(a: int, b: int) -> int:
        log.append(("combine", a + b))
        return a + b

    def finalize(v: int) -> Any:
        log.append(("finalize", v))
        return ("final", v)

    return Plan(process=process, combine=combine, empty=lambda: 0, tasks=tasks, finalize=finalize)


def test_finalize_runs_once_after_the_last_combine() -> None:
    log: list[Any] = []
    res = SequentialRunner().run(_logged_plan(log))
    assert res.value == ("final", 10)
    assert [e for e in log if e[0] == "finalize"] == [("finalize", 10)]
    assert log[-1] == ("finalize", 10) and log[-2] == ("combine", 10)
    assert (res.n_partitions, res.n_combines) == (4, 3)


def test_an_empty_plan_finalizes_its_empty_value() -> None:
    log: list[Any] = []
    res = SequentialRunner().run(_logged_plan(log, tasks=()))
    assert res.value == ("final", 0) and log == [("finalize", 0)]


def test_a_cancelled_run_finalizes_the_fold_it_returns() -> None:
    log: list[Any] = []
    control = RunControl()
    control.cancel()
    res = SequentialRunner(control=control).run(_logged_plan(log))
    assert res.stopped is StopReason.CANCELLED
    assert res.value == ("final", 0) and log == [("finalize", 0)]


def test_without_finalize_the_value_is_the_combined_value() -> None:
    plan = Plan(
        process=lambda p, r: p.entry_start + 1, combine=int.__add__, empty=int, tasks=TASKS
    )
    assert plan.finalize is None
    res = SequentialRunner().run(plan)
    assert res.value == 10
    assert finalize_result(plan, res) is res


def test_a_pickled_plan_keeps_its_finalize() -> None:
    log: list[Any] = []
    plan = cloudpickle.loads(cloudpickle.dumps(_logged_plan(log)))
    assert SequentialRunner().run(plan).value == ("final", 10)


# ---- DurablePlanV2 --------------------------------------------------------------------------


def _reduce_stage(task: Task, inputs: Sequence[bytes], _r: Any) -> bytes:
    return pickle.dumps(sum(pickle.loads(b) for b in inputs))


def _map_stage(task: Task, inputs: Sequence[bytes], _r: Any) -> bytes:
    return pickle.dumps(task.key + 1)


def _staged(finalize: OpSpec | None) -> DurablePlanV2:
    return DurablePlanV2(
        ir=b"finalize-v2",
        stages=(
            StageSpec("map", process=OpSpec("ref", "t:map", live=_map_stage), tasks=TASKS),
            StageSpec(
                "reduce",
                inputs=(0,),
                process=OpSpec("ref", "t:reduce", live=_reduce_stage),
                tasks=(Task(0, Partition("mem://x", "t", 0, 0)),),
            ),
        ),
        finalize=finalize,
    )


def test_a_staged_plan_finalizes_its_value() -> None:
    calls: list[int] = []

    def finalize(v: int) -> int:
        calls.append(v)
        return -v

    assert SequentialRunner().run(_staged(None)).value == 10
    res = SequentialRunner().run(_staged(OpSpec("ref", "t:fin", live=finalize)))
    assert res.value == -10 and calls == [10]


def test_a_cancelled_staged_run_has_no_value_to_finalize() -> None:
    calls: list[int] = []
    control = RunControl()
    control.cancel()
    res = SequentialRunner(control=control).run(_staged(OpSpec("ref", "t:fin", live=calls.append)))
    assert res == ExecResult(None, 0, 0, StopReason.CANCELLED) and calls == []


# ---- serialization --------------------------------------------------------------------------


def _durable(**kw: Any) -> DurablePlan:
    return DurablePlan(
        ir=b"finalize-v1",
        process=OpSpec.from_ref("builtins:len"),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:int"),
        partitions=(Partition("mem://x", "t", 0, 2),),
        **kw,
    )


def test_durable_plans_round_trip_finalize_and_omit_it_when_unset() -> None:
    bare = _durable()
    assert b"finalize" not in bare.to_bytes()
    assert DurablePlan.from_bytes(bare.to_bytes()).finalize is None
    fin = _durable(finalize=OpSpec.from_ref("builtins:str"))
    back = DurablePlan.from_bytes(fin.to_bytes())
    assert back == fin and back.finalize is not None and back.finalize.resolve() is str
    assert back.to_bytes() == fin.to_bytes() != bare.to_bytes()
    part = bare.partitions[0]
    assert fin.task_id(part) == bare.task_id(part)
    assert not fin.opaque
    assert _durable(finalize=OpSpec.from_callable(lambda v: v)).opaque


def test_staged_plans_round_trip_finalize_and_omit_it_when_unset() -> None:
    bare = DurablePlanV2(ir=b"v2")
    assert b"finalize" not in bare.to_bytes()
    assert DurablePlanV2.from_bytes(bare.to_bytes()).finalize is None
    fin = DurablePlanV2(ir=b"v2", finalize=OpSpec.from_ref("builtins:str"))
    assert DurablePlanV2.from_bytes(fin.to_bytes()) == fin
