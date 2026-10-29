"""A plan's default worker backend carries the session backend's behavior (#122).

A behavior method that only the backend registers runs in workers as it runs driver-side, across a
stdlib-pickling process boundary, while a behavior-less session's plan pickles as before.
"""

from __future__ import annotations

import functools
import multiprocessing
import pickle
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import awkward as ak
import cloudpickle
import numpy as np
import pytest

import graphed
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, gak, to_parquet
from graphed.awkward._ops import apply
from graphed.core import Partition, SequentialRunner
from graphed.core.execution import LocalResources, Plan, WorkerResources


class P(ak.Record):  # type: ignore[misc]
    pass


class PArray(ak.Array):  # type: ignore[misc]
    def doubled(self) -> Any:
        return self.x * 2


BEHAVIOR: dict[Any, Any] = {"P": P, ("*", "P"): PArray, "unused": lambda x: x}
EVENTS = ak.Array({"x": np.arange(8.0), "run": np.arange(8, dtype=np.int64) % 3})
LUMI = ak.Array({"run": np.arange(3, dtype=np.int64), "w": np.arange(3.0)})


@dataclass
class Rows:
    data: ak.Array

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind("mem://rows", "", s, steps_per_file) for s in range(steps_per_file)
        )

    def read_partition(
        self, partition: Partition, columns: Any, resources: WorkerResources
    ) -> ak.Array:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def _source(session: Session, name: str, data: ak.Array) -> Array:
    form = AwkwardForm(ak.Array(data.layout.to_typetracer(forget_length=True)))
    return session.source(name, form=form, data=Rows(data))


def _first(values: list[Any]) -> Any:
    return ak.concatenate([values[0], values[0][:0]])  # merges, so leaves a copied behavior


def _cat(a: Any, b: Any) -> Any:
    return ak.concatenate([a, b])


def _empty() -> ak.Array:
    return ak.Array(np.zeros(0))


FOLD: dict[str, Any] = {"reduce": _first, "combine": _cat, "empty": _empty}


def _doubled(x: Array) -> Any:
    return gak.zip({"x": x}, with_name="P").doubled()


def _aggregate(session: Session, out: Any = _doubled, **kwargs: Any) -> Plan[Any]:
    ev = _source(session, "events", EVENTS)
    y = out(ev.x)
    return graphed.aggregate_plan(y, steps_per_file=2, **FOLD, **kwargs)


def _shuffle(session: Session, out: Any = _doubled, **kwargs: Any) -> Any:
    ev = graphed.repartition(_source(session, "events", EVENTS), n=2)
    y = out(ev.x)
    return graphed.shuffle_plan(y, steps_per_file=2, **FOLD, **kwargs)


def _join(session: Session, out: Any = _doubled, **kwargs: Any) -> Any:
    joined = graphed.join(
        _source(session, "events", EVENTS), _source(session, "lumi", LUMI), on=["run"]
    )
    y = out(joined.x + joined.w)
    return graphed.join_plan(y, steps_per_file=2, **FOLD, **kwargs)


def _twice(x: Array) -> Array:
    return x * 2


def _task(process: Callable[[Partition, WorkerResources], Any], partition: Partition) -> Any:
    return process(partition, LocalResources())


def _spawned(plan: Plan[Any]) -> Any:
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(
            pool.map(functools.partial(_task, plan.process), [t.partition for t in plan.tasks])
        )
    return functools.reduce(plan.combine, parts, plan.empty())


def _carries_the_behavior(value: Any, want: ak.Array) -> None:
    assert ak.to_list(value) == ak.to_list(want)
    assert callable(value.behavior["unused"])  # a lambda: stdlib pickle needs the shipping
    assert ak.to_list(pickle.loads(pickle.dumps(value))) == ak.to_list(want)


def test_a_backend_behavior_method_runs_in_spawned_workers() -> None:
    plan = _aggregate(Session(AwkwardBackend(behavior=BEHAVIOR)))
    _carries_the_behavior(_spawned(plan), 2 * EVENTS.x)
    _carries_the_behavior(SequentialRunner().run(plan).value, 2 * EVENTS.x)


@pytest.mark.parametrize("build", [_shuffle, _join], ids=["shuffle", "join"])
def test_a_backend_behavior_method_runs_in_shuffle_and_join_workers(build: Any) -> None:
    plan = pickle.loads(pickle.dumps(build(Session(AwkwardBackend(behavior=BEHAVIOR)))))
    want = EVENTS.x if build is _shuffle else EVENTS.x + LUMI.w[EVENTS.run]
    value = SequentialRunner().run(plan).value
    _carries_the_behavior(value[ak.argsort(value)], ak.sort(2 * want))


@pytest.mark.parametrize(
    "build", [_aggregate, _shuffle, _join], ids=["aggregate", "shuffle", "join"]
)
def test_a_behavior_less_plan_pickles_as_with_the_backend_class(build: Any) -> None:
    def shipped(plan: Any) -> bytes:
        return pickle.dumps(plan.process) if build is _aggregate else bytes(plan.to_bytes())

    default = shipped(build(Session(AwkwardBackend()), out=_twice))
    assert default == shipped(build(Session(AwkwardBackend()), out=_twice, backend=AwkwardBackend))


def test_to_parquet_writes_a_backend_behavior_method_in_workers(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    y = gak.zip({"y": _doubled(_source(session, "events", EVENTS).x)})
    plan = pickle.loads(
        pickle.dumps(to_parquet(y, str(tmp_path), compute=False, steps_per_file=2))
    )
    paths = [
        path for task in plan.tasks for path in plan.process(task.partition, LocalResources())
    ]
    written = ak.concatenate([ak.from_parquet(path) for path in paths])
    assert ak.to_list(written.y) == ak.to_list(2 * EVENTS.x)


def test_a_behavior_that_does_not_pickle_is_refused_when_the_plan_is_built() -> None:
    session = Session(AwkwardBackend(behavior={**BEHAVIOR, "lock": threading.Lock()}))
    with pytest.raises(TypeError, match="backend='module:factory'"):
        _aggregate(session)


def test_an_equal_behavior_is_pickled_once_across_plan_builds(monkeypatch: Any) -> None:
    pickled: list[Any] = []
    dumps = cloudpickle.dumps

    def counting(obj: Any, *args: Any, **kwargs: Any) -> bytes:
        if isinstance(obj, dict):
            pickled.append(obj)
        return bytes(dumps(obj, *args, **kwargs))

    monkeypatch.setattr(cloudpickle, "dumps", counting)
    behavior = {**BEHAVIOR, "fresh": lambda x: x}  # unequal to every behavior shipped before
    plans = [_aggregate(Session(AwkwardBackend(behavior=dict(behavior)))) for _ in range(4)]
    assert len(pickled) == 1
    _carries_the_behavior(_spawned(plans[-1]), 2 * EVENTS.x)

    grown = {**behavior, "more": lambda x: x}
    factory = AwkwardBackend(behavior=grown).worker_factory()
    assert len(pickled) == 2
    assert "more" in factory.keywords["behavior"]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("op", "params", "data"),
    [("neg", {}, np.arange(3.0)), ("ak.with_name", {"name": "P"}, {"x": np.arange(3.0)})],
    ids=["neg", "with_name"],
)
def test_an_operand_carrying_the_backend_behavior_is_not_rewrapped(
    monkeypatch: Any, op: str, params: dict[str, Any], data: Any
) -> None:
    behavior = dict(BEHAVIOR)
    made: list[None] = []
    init = ak.Array.__init__

    def counting(self: ak.Array, *args: Any, **kwargs: Any) -> None:
        made.append(None)
        init(self, *args, **kwargs)

    monkeypatch.setattr(ak.Array, "__init__", counting)

    def built(operand: ak.Array, with_behavior: Any) -> int:
        made.clear()
        result = apply(op, [operand], params, with_behavior)
        if with_behavior is not None:
            assert result.behavior is behavior
        return len(made)

    carrying = ak.Array(data, behavior=behavior)
    assert built(carrying, behavior) == built(carrying, None)
    # an operand without that very object still gets it
    assert built(ak.Array(data, behavior=dict(behavior)), behavior) > built(carrying, None)
    assert built(ak.Array(data), behavior) > built(carrying, None)
