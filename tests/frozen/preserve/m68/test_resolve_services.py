"""m68 §3.2 resolve walk — ``resolve_services(plan, value)`` hands a run's value to the plan's process
when it is ``Resolvable``; ``_PartitionReduce`` forwards it to its ``reduce`` and ``_Collated`` per
name over ``{name: value}``, so a hook anywhere binding reaches is handed its own sub-value."""

from __future__ import annotations

import dataclasses
from typing import Any

from m68_services_fixtures import _cat, _rows, mark, new_events, values_plan

from graphed import aggregate_plan, collate
from graphed.core import Partition, Plan, SequentialRunner, WorkerResources
from graphed.services import ServiceSpec


class _Spy:
    """A ``Resolvable`` part: a process (``inner`` given) or a ``reduce``; records what it resolves."""

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner
        self.calls: list[Any] = []
        self.returns: list[Any] = []

    def __call__(self, *args: Any) -> Any:
        return self.inner(*args) if self.inner is not None else _rows(*args)

    def resolve_services(self, value: Any) -> Any:
        self.calls.append(value)
        self.returns.append(("resolved", object()))
        return self.returns[-1]

    def forget(self) -> None:
        self.calls.clear()
        self.returns.clear()


def _spied(plan: Plan[Any]) -> tuple[Plan[Any], _Spy]:
    spy = _Spy(plan.process)
    return dataclasses.replace(plan, process=spy), spy


def _elsewhere() -> Plan[Any]:
    """An aggregate plan over its own ``(uri, tree)``, so it collates beside the fixtures' source."""
    ev = new_events()
    parts = (Partition.blind("mem://elsewhere", "", 0, 1),)
    return aggregate_plan(ev.x, reduce=_rows, combine=_cat, empty=list, partitions=parts)


def _plain(partition: Partition, resources: WorkerResources) -> list[int]:
    return [partition.entry_stop]


def test_a_resolvable_process_is_handed_the_value() -> None:
    from graphed.services import resolve_services  # noqa: PLC0415

    plan, spy = _spied(values_plan(new_events().x))
    value = SequentialRunner().run(plan).value
    assert value != []
    spy.forget()
    assert resolve_services(plan, value) is spy.returns[0]
    assert len(spy.calls) == 1
    assert spy.calls[0] is value


def test_a_collated_part_is_handed_its_own_sub_value() -> None:
    from graphed.services import resolve_services  # noqa: PLC0415

    spied, spy = _spied(values_plan(new_events().x))
    plan = collate({"a": spied, "b": _elsewhere()})
    value = SequentialRunner().run(plan).value
    assert set(value) == {"a", "b"}
    spy.forget()
    resolved = resolve_services(plan, value)
    assert len(spy.calls) == 1
    assert spy.calls[0] is value["a"]
    assert resolved["a"] is spy.returns[0]
    assert set(resolved) == {"a", "b"}
    assert resolved["b"] is value["b"]


def test_a_resolvable_reduce_is_handed_the_value() -> None:
    from graphed.services import resolve_services  # noqa: PLC0415

    spy = _Spy()
    plan = values_plan(new_events().x, reduce=spy)
    value = SequentialRunner().run(plan).value
    assert value != []
    spy.forget()
    assert resolve_services(plan, value) is spy.returns[0]
    assert len(spy.calls) == 1
    assert spy.calls[0] is value


def test_a_process_without_the_hook_returns_the_value_itself() -> None:
    from graphed.services import resolve_services  # noqa: PLC0415

    plain: Plan[Any] = Plan(process=_plain, combine=_cat, empty=list)
    value = [[1.0, 2.0]]
    assert resolve_services(plain, value) is value
    aggregate = values_plan(new_events().x)  # the reduce has no hook
    ran = SequentialRunner().run(aggregate).value
    assert resolve_services(aggregate, ran) is ran


def test_a_collated_name_without_a_sub_value_is_not_called() -> None:
    from graphed.services import resolve_services  # noqa: PLC0415

    spied, spy = _spied(values_plan(new_events().x))
    plan = collate({"a": dataclasses.replace(spied, tasks=()), "b": _elsewhere()})
    value = SequentialRunner().run(plan).value
    assert set(value) == {"b"}
    resolved = resolve_services(plan, value)
    assert spy.calls == []
    assert resolved == value
    assert resolved["b"] is value["b"]


def test_the_composites_forward_to_the_spy() -> None:
    from graphed.services import Resolvable  # noqa: PLC0415

    spied, part = _spied(values_plan(new_events().x))
    collated = collate({"a": spied, "b": _elsewhere()})
    reduce = _Spy()
    aggregate = values_plan(new_events().x, reduce=reduce)
    assert collated.process is not part
    assert aggregate.process is not reduce
    assert isinstance(collated.process, Resolvable)
    assert isinstance(aggregate.process, Resolvable)
    sub = [[0.5]]
    collated.process.resolve_services({"a": sub})
    assert len(part.calls) == 1
    assert part.calls[0] is sub
    aggregate.process.resolve_services(sub)
    assert len(reduce.calls) == 1
    assert reduce.calls[0] is sub


def test_resolvable_is_runtime_checkable() -> None:
    from graphed.services import Resolvable  # noqa: PLC0415

    assert isinstance(_Spy(), Resolvable)
    assert not isinstance(_plain, Resolvable)
    assert not isinstance(object(), Resolvable)


def test_an_external_evaluator_is_not_resolvable() -> None:  # binds only its endpoint
    from graphed.services import Resolvable  # noqa: PLC0415

    ev = new_events()
    ev.session.declare_service(ServiceSpec("gen-svc", "http"))
    plan = values_plan(mark(ev, "resolve-generic", "gen-svc"))
    ((_key, fn),) = plan.process.externals
    assert hasattr(fn, "bind_services")
    assert not isinstance(fn, Resolvable)
