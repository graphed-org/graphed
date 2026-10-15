"""A plan whose callables are lambdas or closures, and whose partials carry a lambda-holding behavior,
runs across a stdlib-pickling process boundary as it runs in-process (GW33)."""

from __future__ import annotations

import functools
import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.aggregate import aggregate_plan, by_value
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.checkpoint.resume import _key_digest
from graphed.core import Partition
from graphed.core.execution import LocalResources, Plan, SequentialRunner, WorkerResources

EVENTS = ak.Array({"Jet": [[{"pt": float(i % 7 + 1)}] * (i % 3 + 1) for i in range(64)]})
BEHAVIOR: dict[Any, Any] = {"unused": lambda x: x}  # vector's behavior holds lambdas like this


@dataclass
class Events:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def _jet_pt_sums(events: ak.Array) -> ak.Array:
    return ak.sum(events.Jet.pt, axis=1)


def make_backend() -> AwkwardBackend:
    return AwkwardBackend(behavior=BEHAVIOR)


def _plan(k: float, backend: str | None = None) -> Plan[Any]:
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    ev = session.source("events", form=form, data=Events(EVENTS))
    pt = graphed.apply(lambda x: x * k, ev.Jet.pt)  # a closure over k
    return aggregate_plan(
        gak.sum(pt, axis=1),
        reduce=lambda values: ak.Array(values[0]),
        combine=lambda a, b: ak.concatenate([a, b]),
        empty=lambda: ak.Array([]),
        steps_per_file=4,
        backend=backend,
    )


def _task(process: Callable[[Partition, WorkerResources], Any], partition: Partition) -> Any:
    return process(partition, LocalResources())


def _spawned(plan: Plan[Any]) -> Any:
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(functools.partial(_task, plan.process), [t.partition for t in plan.tasks]))
    return functools.reduce(plan.combine, parts, plan.empty())


@pytest.mark.parametrize("backend", [None, f"{__name__}:make_backend"], ids=["class", "import_ref"])
def test_a_plan_of_closures_runs_in_spawned_workers_as_in_process(backend: str | None) -> None:
    plan = _plan(3.0, backend)
    got = _spawned(plan)
    assert ak.to_list(got) == ak.to_list(SequentialRunner().run(plan).value)
    assert ak.sum(got) == 3.0 * ak.sum(_jet_pt_sums(EVENTS))


def test_a_partial_carrying_the_backend_behavior_pickles_by_its_import_ref() -> None:
    plan = _plan(2.0, f"{__name__}:make_backend")
    partial = plan.process(plan.tasks[0].partition, LocalResources())
    assert partial.behavior is not None and "unused" in partial.behavior
    assert ak.to_list(pickle.loads(pickle.dumps(partial))) == ak.to_list(partial)


def test_wrapping_a_closure_leaves_its_task_key_unchanged() -> None:
    k = 2.0
    closure = lambda x: x * k  # noqa: E731
    wrapped = by_value(closure)
    assert wrapped is not closure
    assert _key_digest(wrapped) == _key_digest(closure)
    assert pickle.loads(pickle.dumps(wrapped))(3.0) == 6.0
    assert by_value(_jet_pt_sums) is _jet_pt_sums


def test_resolving_a_backend_from_its_import_ref_leaves_its_key_unchanged() -> None:
    resolved = make_backend().resolved_from(f"{__name__}:make_backend")
    assert type(resolved._behavior).__name__ == "_BehaviorByRef"
    assert _key_digest(resolved) == _key_digest(make_backend())
