"""A service as data: the spec a plan carries and the protocols its parts bind through.

:mod:`graphed.services` re-exports all of it beside the functions that bind a plan, which need
the plan types this module sits below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from ..errors import GraphedError

#: The wires an endpoint may name; the ``s`` schemes are TLS.
SCHEMES = ("tcp", "http", "https", "grpc", "grpcs")


def _read_only(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class Launch:
    """How to start a managed instance.

    ``argv`` templates (``{port}``, ``{host}``, ``{python}``), an optional ``image`` (a cvmfs path
    or registry ref), staged ``inputs``, ``env`` and ``resources``
    (``cpus``/``memory_mb``/``gpus``).
    """

    argv: tuple[str, ...]
    image: str | None = None
    inputs: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    resources: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Freeze the argument and mapping fields."""
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "env", _read_only(self.env))
        object.__setattr__(self, "resources", _read_only(self.resources))

    def __reduce__(self) -> tuple[Any, ...]:
        # a mappingproxy does not pickle; the constructor re-wraps the plain dicts
        """Pickle with the mappings as plain dicts."""
        return (Launch, (self.argv, self.image, self.inputs, dict(self.env), dict(self.resources)))

    def __hash__(self) -> int:  # a mappingproxy does not hash
        """Hash the fields, the mappings by their items."""
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
        """Return the launch as a JSON-ready dict."""
        return {
            "argv": list(self.argv),
            "env": dict(sorted(self.env.items())),
            "image": self.image,
            "inputs": list(self.inputs),
            "resources": dict(sorted(self.resources.items())),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Launch:
        """Read a launch :meth:`to_json` wrote."""
        return cls(**doc)


@dataclass(frozen=True)
class ServiceSpec:
    """A service an analysis needs, by ``name``.

    ``launch`` is how to start one when no endpoint is given (``None``: an external service only).
    """

    name: str
    kind: str
    check: str = "tcp"
    ports: tuple[int, int] = (10000, 10100)
    launch: Launch | None = None
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        """Normalize ``ports`` to a pair of ints."""
        low, high = self.ports
        object.__setattr__(self, "ports", (int(low), int(high)))

    def to_json(self) -> dict[str, Any]:
        """Return the spec as a JSON-ready dict."""
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
        """Read a spec :meth:`to_json` wrote."""
        launch = doc["launch"]
        return cls(**{**doc, "launch": None if launch is None else Launch.from_json(launch)})


class UnboundServiceError(GraphedError):
    """Nodes name services no endpoint was bound for.

    ``names`` lists them, ``name`` is the first.
    """

    def __init__(self, *names: str) -> None:
        """Name the services that have no endpoint, with how to bind them."""
        self.names = names
        self.name = names[0]
        listed = ", ".join(map(repr, names))
        endpoints = ", ".join(f"{name!r}: 'scheme://host:port'" for name in names)
        has = f"service {listed} has" if len(names) == 1 else f"services {listed} have"
        super().__init__(
            f"{has} no endpoint: bind with graphed.services.bind_services(plan, {{{endpoints}}}), "
            "or run the plan through an executor that resolves Plan.services"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by the service names."""
        return (UnboundServiceError, self.names)


def split_endpoint(endpoint: str) -> tuple[str, str]:
    """``"grpcs://host:443"`` -> ``("grpcs", "host:443")``.

    Anything but ``scheme://host:port`` with a scheme in :data:`SCHEMES` is refused.
    """
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
            f"an endpoint is scheme://host:port with a scheme in {', '.join(SCHEMES)}; got "
            f"{endpoint!r}"
        )
    return parts.scheme, parts.netloc


@runtime_checkable
class Bindable(Protocol):
    """A plan part that takes run endpoints: returns a bound copy, never binds in place.

    A part that calls a service raises :class:`UnboundServiceError` naming it when ``endpoints``
    lacks the name and the part holds no endpoint of its own, so :func:`require_bound` sees every
    service a part needs.
    """

    def bind_services(self, endpoints: Mapping[str, str]) -> Any:
        """Return this part with ``endpoints`` bound in."""
        ...


@runtime_checkable
class Resolvable(Protocol):
    """A plan part that turns a run's value into its final form while services are up.

    A receipt into a snapshot, say; it returns the resolved value, never resolves in place.
    """

    def resolve_services(self, value: Any) -> Any:
        """Return ``value`` in its final form."""
        ...
