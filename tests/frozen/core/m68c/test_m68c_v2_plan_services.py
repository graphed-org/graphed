"""m68c §3.1/§3.2: a ``DurablePlanV2`` carries ``services`` outside its identity, and binding an
endpoint into its stage processes changes neither its bytes nor any task id."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import pytest

from graphed.core import DurablePlanV2, GraphStore, OpSpec, Partition, StageSpec, Task
from graphed.services import ServiceSpec, UnboundService, bind_services

SPEC = ServiceSpec("sf", "http")
URL = "http://sf.example:8080"


@dataclass(frozen=True)
class Stamp:
    """A bindable stage process whose payload names the endpoint it ran with."""

    endpoint: str | None = None

    def __call__(self, task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
        if self.endpoint is None:
            raise UnboundService("sf")
        return pickle.dumps((task.key, self.endpoint))

    def bind_services(self, endpoints: Mapping[str, str]) -> Stamp:
        endpoint = endpoints.get("sf", self.endpoint)
        if endpoint is None:
            raise UnboundService("sf")
        return replace(self, endpoint=endpoint)


def gather(task: Task, inputs: tuple[bytes, ...], resources: Any) -> bytes:
    return pickle.dumps(sorted(pickle.loads(p) for p in inputs))


def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "mem://events"})
    out = g.add_exchange([g.add_op("pt", [src])], {"scheme": "count", "parts": 2})
    return bytes(g.serialize(outputs=[out]))


def _stages() -> tuple[StageSpec, ...]:
    return (
        StageSpec(
            kind="map_write",
            process=OpSpec.from_callable(Stamp()),
            routing={"scheme": "count", "parts": 2, "backend_id": "toy/0"},
            tasks=tuple(Task(i, Partition("mem://events", "", i, i + 1)) for i in range(2)),
        ),
        StageSpec(
            kind="gather",
            inputs=(0,),
            process=OpSpec.from_callable(gather),
            routing={"parts": 2, "backend_id": "toy/0"},
            tasks=tuple(Task(d, Partition("dest", "p", d, d + 1)) for d in range(2)),
        ),
    )


def _task_ids(plan: DurablePlanV2) -> list[str]:
    return [plan.task_id(si, t) for si, stage in enumerate(plan.stages) for t in stage.tasks]


def test_services_round_trip_and_stay_out_of_the_bytes_when_empty_and_out_of_task_ids() -> None:
    served = DurablePlanV2(ir=_ir(), stages=_stages(), services=(SPEC,))
    plain = DurablePlanV2(ir=_ir(), stages=_stages())

    again = DurablePlanV2.from_bytes(served.to_bytes())
    assert again.services == (SPEC,)
    assert again.to_bytes() == served.to_bytes()
    assert json.loads(served.to_bytes())["services"] == [SPEC.to_json()]

    assert set(json.loads(plain.to_bytes())) == {"format_version", "ir_b64", "stages"}
    assert DurablePlanV2.from_bytes(plain.to_bytes()).services == ()
    assert _task_ids(served) == _task_ids(plain)
    assert len(set(_task_ids(plain))) == 4


def test_binding_keeps_bytes_and_task_ids_and_reaches_the_bindable_stage_only() -> None:
    plan = DurablePlanV2(ir=_ir(), stages=_stages())
    bound = bind_services(plan, {"sf": URL})

    assert bound.to_bytes() == plan.to_bytes()
    assert _task_ids(bound) == _task_ids(plan)
    bound_stamp, stamp = bound.stages[0].process.resolve(), plan.stages[0].process.resolve()
    assert isinstance(bound_stamp, Stamp)
    assert isinstance(stamp, Stamp)
    assert (bound_stamp.endpoint, stamp.endpoint) == (URL, None)
    assert bound.stages[1].process.resolve() is gather
    with pytest.raises(ValueError, match="scheme://host:port"):
        bind_services(plan, {"sf": "sf.example:8080"})
