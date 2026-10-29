"""An analysis's service surface: the external services its nodes call, declared as data.

A :class:`ServiceSpec` is the requirement (``name`` nodes reference, ``kind`` a site table matches,
``check`` the readiness rule, ``ports`` a managed instance may bind) plus an optional
:class:`Launch` recipe. A ``Session`` holds the declared specs (``declare_service``), a node names
one through ``params["service"]``, and ``Plan``/``DurablePlan``/the preservation bundle carry the
specs the graph references. The endpoint a run reaches a service at is environment, never graph
identity:
:func:`bind_services` hands it to a plan's process after the IR is fixed, :func:`resolve_services`
hands the run's value back through the same parts, and ``RunReport.endpoints`` records it as run
provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar, overload

from .core.execution import _stage_processes, require_bound
from .core.plan import DurablePlanV2, OpSpec
from .core.services import (
    SCHEMES,
    Bindable,
    Launch,
    Resolvable,
    ServiceSpec,
    UnboundServiceError,
    split_endpoint,
)

if TYPE_CHECKING:
    from .core.execution import Plan
    from .session import Session

R = TypeVar("R")

# The pre-0.0.10 name, kept for graphed-executors and graphed-histogram, which still import it.
UnboundService = UnboundServiceError


def referenced_services(
    session: Session, nodes: Iterable[Mapping[str, Any]], names: Iterable[str] | None = None
) -> tuple[ServiceSpec, ...]:
    """Return the ``session``'s specs the External ``nodes`` and ``names`` name.

    The nodes name them by ``params["service"]``; in name order; an undeclared name is refused.
    """
    named = {
        str(n["params"]["service"])
        for n in nodes
        if n["kind"] == "external" and "service" in n["params"]
    }
    return tuple(session.service_for(name) for name in sorted(named.union(names or ())))


def bind_externals(
    externals: Iterable[tuple[str, Any]], endpoints: Mapping[str, str]
) -> tuple[tuple[str, Any], ...]:
    """Return a plan process's ``(key, evaluator)`` pairs.

    Each :class:`Bindable` evaluator is bound.
    """
    return tuple(
        (key, fn.bind_services(endpoints) if isinstance(fn, Bindable) else fn)
        for key, fn in externals
    )


@overload
def bind_services(plan: DurablePlanV2, endpoints: Mapping[str, str]) -> DurablePlanV2: ...
@overload
def bind_services(plan: Plan[R], endpoints: Mapping[str, str]) -> Plan[R]: ...
def bind_services(
    plan: Plan[R] | DurablePlanV2, endpoints: Mapping[str, str]
) -> Plan[R] | DurablePlanV2:
    """Return ``plan`` with ``endpoints`` bound into its process.

    ``endpoints`` maps service name -> ``scheme://host:port``; a ``DurablePlanV2`` binds into each
    stage process, as ``OpSpec.live``, so bytes and task ids stay. The same plan when nothing has a
    ``bind_services`` hook. Every endpoint is checked first.
    """
    for endpoint in endpoints.values():
        split_endpoint(endpoint)
    if isinstance(plan, DurablePlanV2):
        stages = []
        for stage in plan.stages:
            op, fn = stage.process, stage.process.resolve()
            if isinstance(fn, Bindable):
                stage = replace(
                    stage,
                    process=OpSpec(op.kind, op.ref, op.blob_b64, live=fn.bind_services(endpoints)),
                )
            stages.append(stage)
        return replace(plan, stages=tuple(stages))
    if not isinstance(plan.process, Bindable):
        return plan
    return replace(plan, process=plan.process.bind_services(endpoints))


@overload
def resolve_services(plan: DurablePlanV2, value: Any) -> Any: ...
@overload
def resolve_services(plan: Plan[R], value: R) -> R: ...
def resolve_services(plan: Plan[R] | DurablePlanV2, value: Any) -> Any:
    """Return ``value``, a run of ``plan``'s, resolved through its process.

    A ``DurablePlanV2`` resolves through each stage process in stage order; ``value`` itself when
    nothing has a ``resolve_services`` hook. A runner that holds the run's services calls it at the
    end of the run, before they close (``SequentialRunner`` holds none and does not); it walks the
    parts :func:`bind_services` reaches.
    """
    parts = _stage_processes(plan) if isinstance(plan, DurablePlanV2) else (plan.process,)
    for part in parts:
        if isinstance(part, Resolvable):
            value = part.resolve_services(value)
    return value


__all__ = [
    "SCHEMES",
    "Bindable",
    "Launch",
    "Resolvable",
    "ServiceSpec",
    "UnboundServiceError",
    "bind_externals",
    "bind_services",
    "referenced_services",
    "require_bound",
    "resolve_services",
    "split_endpoint",
]
