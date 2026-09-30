"""m68c §3.4: ``SequentialRunner.run`` runs a ``DurablePlanV2`` stage by stage with full upstream
payloads, returns ``plan.value`` of the last stage, refuses an unbound plan before any process call,
and follows V1's run control (cancel on entry, ``wait()`` before each stage, ``reset()`` at the end)."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import pytest

from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task
from graphed.core.execution import LocalResources, RunControl, RunState, SequentialRunner, StopReason
from graphed.services import ServiceSpec, UnboundService, bind_services

#: every stage-process call, as ``(stage name, task key)``
CALLS: list[tuple[str, int]] = []
#: the resources object each call was handed
SEEN: list[object] = []
#: the control :func:`cancelling_emit` cancels
CONTROL: list[RunControl] = []


def emit(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    CALLS.append(("map", task.key))
    SEEN.append(resources)
    return pickle.dumps(task.key + 1)


def cancelling_emit(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    CONTROL[0].cancel()
    return emit(task, inputs, resources)


def scale(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    CALLS.append(("gather", task.key))
    SEEN.append(resources)
    return pickle.dumps(sum(pickle.loads(p) for p in inputs) * (task.key + 1))


def total(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    CALLS.append(("reduce", task.key))
    SEEN.append(resources)
    return pickle.dumps(sum(pickle.loads(p) for p in inputs))


@dataclass(frozen=True)
class Served:
    """A bindable map process that needs the ``sf`` endpoint."""

    endpoint: str | None = None

    def __call__(self, task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
        CALLS.append(("served", task.key))
        return pickle.dumps(f"{self.endpoint}#{task.key}")

    def bind_services(self, endpoints: Mapping[str, str]) -> Served:
        endpoint = endpoints.get("sf", self.endpoint)
        if endpoint is None:
            raise UnboundService("sf")
        return replace(self, endpoint=endpoint)


def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "mem://events"})
    out = g.add_exchange([g.add_op("pt", [src])], {"scheme": "count", "parts": 2})
    return bytes(g.serialize(outputs=[out]))


def _stage(kind: str, fn: Any, inputs: tuple[int, ...], n: int) -> StageSpec:
    return StageSpec(
        kind=kind,
        inputs=inputs,
        process=OpSpec.from_callable(fn),
        routing={"parts": 2, "backend_id": "toy/0"},
        tasks=tuple(Task(k, Partition("mem://events", "", k, k + 1)) for k in range(n)),
    )


def _plan(first: Any = emit, *, fold: bool = True) -> DurablePlanV2:
    stages = [_stage("map_write", first, (), 2), _stage("gather", scale, (0,), 2)]
    if fold:
        stages.append(_stage("reduce", total, (1,), 1))
    return DurablePlanV2(ir=_ir(), stages=tuple(stages))


@pytest.fixture(autouse=True)
def _clear() -> None:
    CALLS.clear()
    SEEN.clear()
    CONTROL.clear()


def test_a_completed_run_returns_the_last_stages_value_in_stage_order() -> None:
    res = SequentialRunner().run(_plan())
    assert res.value == 9
    assert res.stopped is None
    assert (res.n_partitions, res.n_combines) == (2, 3)
    assert CALLS == [("map", 0), ("map", 1), ("gather", 0), ("gather", 1), ("reduce", 0)]
    assert len({id(r) for r in SEEN}) == 1
    assert isinstance(SEEN[0], LocalResources)

    CALLS.clear()
    unfolded = SequentialRunner().run(_plan(fold=False))
    assert unfolded.value == (3, 6)
    assert (unfolded.n_partitions, unfolded.n_combines) == (2, 2)


def test_an_unbound_plan_is_refused_before_any_process_call_and_a_bound_one_runs() -> None:
    stages = (_stage("map_write", Served(), (), 2),)
    plan = DurablePlanV2(ir=_ir(), stages=stages, services=(ServiceSpec("sf", "http"),))
    with pytest.raises(UnboundService, match="'sf'"):
        SequentialRunner().run(plan)
    assert CALLS == []

    res = SequentialRunner().run(bind_services(plan, {"sf": "http://sf.example:8080"}))
    assert res.value == ("http://sf.example:8080#0", "http://sf.example:8080#1")


def test_a_control_cancelled_before_run_stops_with_no_process_call() -> None:
    control = RunControl()
    control.cancel()
    res = SequentialRunner(control=control).run(_plan())
    assert res.stopped is StopReason.CANCELLED
    assert res.value is None
    assert (res.n_partitions, res.n_combines) == (0, 0)
    assert CALLS == []
    assert control.state is RunState.RUNNING


def test_a_cancel_inside_the_first_stage_stops_before_the_next_stage() -> None:
    control = RunControl()
    CONTROL.append(control)
    res = SequentialRunner(control=control).run(_plan(cancelling_emit))
    assert res.stopped is StopReason.CANCELLED
    assert res.value is None
    ran = [c for c in CALLS if c[0] == "map"]
    assert ran
    assert ran == CALLS
    assert (res.n_partitions, res.n_combines) == (len(ran), 0)
    assert control.state is RunState.RUNNING
