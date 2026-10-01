"""An analysis's service surface: the external services its nodes call, declared as data.

A :class:`ServiceSpec` is the requirement (``name`` nodes reference, ``kind`` a site table matches,
``check`` the readiness rule, ``ports`` a managed instance may bind) plus an optional
:class:`Launch` recipe. A ``Session`` holds the declared specs (``declare_service``), a node names one
through ``params["service"]``, and ``Plan``/``DurablePlan``/the preservation bundle carry the specs
the graph references. The endpoint a run reaches a service at is environment, never graph identity:
:func:`bind_services` hands it to a plan's process after the IR is fixed, :func:`resolve_services`
hands the run's value back through the same parts, and ``RunReport.endpoints`` records it as run
provenance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload, runtime_checkable
from urllib.parse import urlsplit

from .errors import GraphedError

if TYPE_CHECKING:
    from .core.execution import Plan
    from .core.plan import DurablePlanV2
    from .session import Session

R = TypeVar("R")

#: The wires an endpoint may name; the ``s`` schemes are TLS.
SCHEMES = ("tcp", "http", "https", "grpc", "grpcs")


def _read_only(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class Launch:
    """How to start a managed instance: ``argv`` templates (``{port}``, ``{host}``, ``{python}``),
    an optional ``image`` (a cvmfs path or registry ref), staged ``inputs``, ``env`` and
    ``resources`` (``cpus``/``memory_mb``/``gpus``)."""

    argv: tuple[str, ...]
    image: str | None = None
    inputs: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    resources: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "env", _read_only(self.env))
        object.__setattr__(self, "resources", _read_only(self.resources))

    def __reduce__(self) -> tuple[Any, ...]:
        # a mappingproxy does not pickle; the constructor re-wraps the plain dicts
        return (Launch, (self.argv, self.image, self.inputs, dict(self.env), dict(self.resources)))

    def __hash__(self) -> int:  # a mappingproxy does not hash
        return hash(
            (
                self.argv,
                self.image,
                self.inputs,
                frozenset(self.env.items()),
                frozenset(self.resources.items()),
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "env": dict(sorted(self.env.items())),
            "image": self.image,
            "inputs": list(self.inputs),
            "resources": dict(sorted(self.resources.items())),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Launch:
        return cls(**doc)


@dataclass(frozen=True)
class ServiceSpec:
    """A service an analysis needs, by ``name``; ``launch`` is how to start one when no endpoint is
    given (``None``: an external service only)."""

    name: str
    kind: str
    check: str = "tcp"
    ports: tuple[int, int] = (10000, 10100)
    launch: Launch | None = None
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        low, high = self.ports
        object.__setattr__(self, "ports", (int(low), int(high)))

    def to_json(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "kind": self.kind,
            "launch": None if self.launch is None else self.launch.to_json(),
            "name": self.name,
            "ports": list(self.ports),
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ServiceSpec:
        launch = doc["launch"]
        return cls(**{**doc, "launch": None if launch is None else Launch.from_json(launch)})


class UnboundService(GraphedError):
    """Nodes name services no endpoint was bound for; ``names`` lists them, ``name`` is the first."""

    def __init__(self, *names: str) -> None:
        self.names = names
        self.name = names[0]
        listed = ", ".join(map(repr, names))
        endpoints = ", ".join(f"{name!r}: 'scheme://host:port'" for name in names)
        has = f"service {listed} has" if len(names) == 1 else f"services {listed} have"
        super().__init__(
            f"{has} no endpoint: bind with graphed.services.bind_services(plan, {{{endpoints}}}), or run"
            " the plan through an executor that resolves Plan.services"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (UnboundService, self.names)


def split_endpoint(endpoint: str) -> tuple[str, str]:
    """``"grpcs://host:443"`` -> ``("grpcs", "host:443")``; anything but ``scheme://host:port`` with a
    scheme in :data:`SCHEMES` is refused."""
    parts = urlsplit(endpoint)
    try:
        port = parts.port
    except ValueError:
        port = None
    if (
        parts.scheme not in SCHEMES
        or port is None
        or not parts.hostname
        or "@" in parts.netloc
        or endpoint != f"{parts.scheme}://{parts.netloc}"
    ):
        raise ValueError(
            f"an endpoint is scheme://host:port with a scheme in {', '.join(SCHEMES)}; got {endpoint!r}"
        )
    return parts.scheme, parts.netloc


@runtime_checkable
class Bindable(Protocol):
    """A plan part that takes run endpoints: returns a bound copy, never binds in place. A part that
    calls a service raises :class:`UnboundService` naming it when ``endpoints`` lacks the name and the
    part holds no endpoint of its own, so :func:`require_bound` sees every service a part needs."""

    def bind_services(self, endpoints: Mapping[str, str]) -> Any: ...


@runtime_checkable
class Resolvable(Protocol):
    """A plan part that turns a run's value into its final form while the run's services are up
    (a receipt into a snapshot, say): returns the resolved value, never resolves in place."""

    def resolve_services(self, value: Any) -> Any: ...


def referenced_services(
    session: Session, nodes: Iterable[Mapping[str, Any]], names: Iterable[str] | None = None
) -> tuple[ServiceSpec, ...]:
    """The ``session``'s specs named by the External ``nodes``' ``params["service"]`` and by
    ``names``, in name order; an undeclared name is refused."""
    named = {
        str(n["params"]["service"]) for n in nodes if n["kind"] == "external" and "service" in n["params"]
    }
    return tuple(session.service_for(name) for name in sorted(named.union(names or ())))


def bind_externals(
    externals: Iterable[tuple[str, Any]], endpoints: Mapping[str, str]
) -> tuple[tuple[str, Any], ...]:
    """A plan process's ``(key, evaluator)`` pairs with each :class:`Bindable` evaluator bound."""
    return tuple(
        (key, fn.bind_services(endpoints) if isinstance(fn, Bindable) else fn) for key, fn in externals
    )


def _stage_processes(plan: DurablePlanV2) -> tuple[Any, ...]:
    return tuple(stage.process.resolve() for stage in plan.stages)


def require_bound(plan: Plan[Any] | DurablePlanV2) -> None:
    """Raise :class:`UnboundService` naming every service ``plan``'s process (a ``DurablePlanV2``'s
    stage processes) has no endpoint for; a runner calls it before its first task. A plan without
    ``services`` returns at once."""
    if not plan.services:
        return
    from .core.plan import DurablePlanV2  # noqa: PLC0415  (core.plan imports this module)

    parts = _stage_processes(plan) if isinstance(plan, DurablePlanV2) else (plan.process,)
    _require_parts([p for p in parts if isinstance(p, Bindable)])


def _require_parts(parts: Sequence[Bindable]) -> None:
    missing: dict[str, str] = {}
    while True:  # each pass through bind_services' own traversal names one more unbound service
        try:
            for part in parts:
                part.bind_services(missing)
        except UnboundService as err:
            if err.name in missing:  # a part refusing a name it was handed breaks the Bindable contract
                raise
            missing[err.name] = "tcp://unbound:0"
            continue
        if missing:
            raise UnboundService(*sorted(missing))
        return


@overload
def bind_services(plan: DurablePlanV2, endpoints: Mapping[str, str]) -> DurablePlanV2: ...
@overload
def bind_services(plan: Plan[R], endpoints: Mapping[str, str]) -> Plan[R]: ...
def bind_services(plan: Plan[R] | DurablePlanV2, endpoints: Mapping[str, str]) -> Plan[R] | DurablePlanV2:
    """``plan`` with ``endpoints`` (service name -> ``scheme://host:port``) bound into its process (a
    ``DurablePlanV2``: into each stage process, as ``OpSpec.live``, so bytes and task ids stay); the
    same plan when nothing has a ``bind_services`` hook. Every endpoint is checked first."""
    for endpoint in endpoints.values():
        split_endpoint(endpoint)
    from .core.plan import DurablePlanV2, OpSpec  # noqa: PLC0415  (core.plan imports this module)

    if isinstance(plan, DurablePlanV2):
        stages = []
        for stage in plan.stages:
            op, fn = stage.process, stage.process.resolve()
            if isinstance(fn, Bindable):
                stage = replace(
                    stage, process=OpSpec(op.kind, op.ref, op.blob_b64, live=fn.bind_services(endpoints))
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
    """``value``, a run of ``plan``'s, resolved through its process (a ``DurablePlanV2``: through each
    stage process in stage order); ``value`` itself when nothing has a ``resolve_services`` hook. A
    runner that holds the run's services calls it at the end of the run, before they close
    (``SequentialRunner`` holds none and does not); it walks the parts :func:`bind_services` reaches."""
    from .core.plan import DurablePlanV2  # noqa: PLC0415  (core.plan imports this module)

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
    "UnboundService",
    "bind_externals",
    "bind_services",
    "referenced_services",
    "require_bound",
    "resolve_services",
    "split_endpoint",
]
