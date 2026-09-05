"""`Varied`: the §2.2 container of universes, and the §2.4 label-aligned combination rule.

A `Varied` holds `{label: member}` with `"nominal"` always first. It is a plain frontend object —
no IR type, no NodeKey (§3.1) — that carries the whole public `Array` surface by mapping each call
over its members, so an analysis written against `Array` records every universe by construction.

The container is PER IDIOM, mirroring `Session._array_cls`: the neutral base carries `Array`'s
surface and an idiom package registers a subclass carrying its own (`graphed.numpy`'s properties
and method set). The idiom comes from the MEMBERS, never from the container.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from ._points import Point, restrict
from .array import Array
from .errors import GraphedError

#: §2.2's reserved names: resolving them as label-mapped field access would let
#: `compile_ir(session, varied)` — which reads `arr.node_id` per output — silently compile.
RESERVED = ("node_id", "session")

Member = Any  # an `Array`, or a `Varied` when the container is a registered weight factor (§2.1)


class Varied:
    """A mapping of universes, `"nominal"` first (§2.2)."""

    __slots__ = ("__weakref__", "_context", "_members", "_tags")

    #: lets `Array._binary` defer to the reflected dunders below instead of raising on a container
    _varied_container = True
    #: a broadcast `__eq__` records an op rather than answering a bool, exactly as `Array` does
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        members: Mapping[str, Member],
        *,
        tags: Mapping[str, tuple[str, ...]] | None = None,
        context: object = None,
    ) -> None:
        if "nominal" not in members:
            raise GraphedError("a Varied must carry a 'nominal' universe")
        self._members: dict[str, Member] = dict(members)
        self._tags: dict[str, tuple[str, ...]] = dict(tags or {})
        self._context = context

    # ---- the §2.2 mapping ------------------------------------------------------
    def _universe(self, label: str) -> Member:
        try:
            return self._members[label]
        except KeyError:
            raise KeyError(
                f"unknown variation label {label!r}; this container carries {list(self._members)}"
            ) from None

    def _member_for(self, label: str) -> Member:
        """§4.6: the member whose point equals `restrict(point(L), axes(self))`, else the central
        universe.

        The fast path — a label this container carries — is not a special case but a theorem:
        `keys(point(L)) ⊆ axes(C)` makes the restriction the identity and point→label is unique
        (§4.11-2). Only the FALLBACK branch projects, and on a container all of whose labels carry
        default points it lands on today's answer (§4.7's theorem), since the only label whose
        point is `{n: t}` is `f"{n}_{t}"`, which the fast path would already have returned.
        """
        member = self._members.get(label)
        if member is not None:
            return member
        point = point_registry(self).get(label)
        if point is not None:
            carried = registered_points(self)
            wanted = restrict(point, frozenset(n for own in carried.values() for n, _ in own))
            if wanted:
                for own_label, own_point in carried.items():
                    if own_point == wanted:
                        return self._members[own_label]
        return self._members["nominal"]

    def apply(self, fn: Callable[[Any], Any]) -> Varied:
        """Apply a record-time ``Array -> Array`` function per universe (§2.2).

        Named `apply`, not `map`: `Array.map` is an EXECUTION-time data callable, and the two
        contracts must not share a name.
        """
        out: dict[str, Member] = {}
        for label, member in self._members.items():
            result = fn(member)
            if isinstance(result, Varied):
                raise GraphedError(
                    "Varied.apply must return one Array per universe; combine two containers with "
                    "ordinary ops instead, which align them label by label"
                )
            out[label] = result
        return rebuild(out, tags=self._tags, context=self._context)

    # ---- structural access -----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in RESERVED:
            raise AttributeError(
                f"{name!r} is not defined on a Varied; read one universe first, with "
                "graphed.universe(v, label) or graphed.nominal(v)"
            )
        if name.startswith("_"):
            raise AttributeError(name)
        return expand(lambda member: getattr(member, name), (self,), {})

    def __iter__(self) -> Iterator[Any]:
        raise TypeError(
            "deferred graphed arrays are not iterable (unknown partitioned length); materialize first"
        )

    def __repr__(self) -> str:
        return f"Varied(labels={list(self._members)})"


# ---- §4.5's Session label -> point registry, read through the nominal member ----------------
def session_of(value: object) -> Any:
    """The Session `value` records into, `None` when it is not a recorded value.

    Reached through the nominal member because §2.2 RESERVES `session` on a `Varied` — resolution
    has no other legal spelling (§4.5).
    """
    while isinstance(value, Varied):
        value = value._members["nominal"]
    return getattr(value, "session", None)


def point_registry(value: object) -> Mapping[str, Point]:
    """The `{label: point}` registry of the Session `value` records into (§4.5)."""
    session = session_of(value)
    return {} if session is None else session._points


def registered_points(container: Varied) -> dict[str, Point]:
    """The registered point of each label `container` carries, in its label order.

    A label with no entry is ABSENT rather than mapped to the origin: it resolves by exact match,
    which is today's rule and what keeps `graphed.varied.rebuild` usable as a hand-built escape
    hatch (§4.5). `"nominal"` is one such label on every container.
    """
    registry = point_registry(container)
    return {label: registry[label] for label in container._members if label in registry}


# ---- the §2.4 combination rule ------------------------------------------------------------
def labels_of(value: object) -> tuple[str, ...]:
    return tuple(value._members) if isinstance(value, Varied) else ()


def member_of(value: object, label: str) -> Any:
    """The universe `value` contributes for `label` (§2.4): its own member, else its nominal one."""
    return value._member_for(label) if isinstance(value, Varied) else value


def universes_of(value: object) -> tuple[Any, ...]:
    """Every universe `value` stands for: its members, or itself when it is not a container."""
    return tuple(value._members.values()) if isinstance(value, Varied) else (value,)


def _collect(value: object, found: list[Varied]) -> None:
    """Every `Varied` in an operand, INSIDE Mapping/Sequence arguments too (§2.3c's
    *container-traversing* class: `gak.zip`'s mapping is its only array-bearing operand)."""
    if isinstance(value, Varied):
        found.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect(item, found)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            _collect(item, found)


def narrow(value: object, label: str) -> Any:
    """`value` narrowed to one label, INSIDE Mapping/Sequence arguments too."""
    if isinstance(value, Varied):
        return value._member_for(label)
    if isinstance(value, Mapping):
        return {key: narrow(item, label) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return type(value)(narrow(item, label) for item in value)  # type: ignore[call-arg]
    return value


def containers_in(*values: object) -> list[Varied]:
    found: list[Varied] = []
    for value in values:
        _collect(value, found)
    return found


def union_labels(*values: object) -> tuple[str, ...]:
    """§2.4's bound union ORDER: the first operand's labels, then labels new to the next, and so
    on, `"nominal"` always first. §3.2's determinism gate and §6.1c's positional layout both
    depend on this being insertion ordered, never set ordered."""
    out: dict[str, None] = {"nominal": None}
    for container in containers_in(*values):
        for label in container._members:
            out.setdefault(label, None)
    return tuple(out)


def union_tags(*values: object) -> dict[str, tuple[str, ...]]:
    """The per-`name` tag map §1.1's family check reads, carried through combinations."""
    merged: dict[str, tuple[str, ...]] = {}
    for container in containers_in(*values):
        for name, tags in container._tags.items():
            kept = merged.get(name, ())
            merged[name] = kept + tuple(t for t in tags if t not in kept)
    return merged


def most_derived_context(*values: Any) -> object:
    """§2.3e's container handle: the most-derived member handle, `None` when all are context-free."""
    from .accessors import context_of, unify_contexts  # noqa: PLC0415  (import cycle)

    return unify_contexts(*(context_of(value) for value in values))


def rebuild(
    members: Mapping[str, Member],
    *,
    tags: Mapping[str, tuple[str, ...]] | None = None,
    context: object = None,
) -> Varied:
    """Build the container class PAIRED WITH the members' idiom (§2.2)."""
    central = members["nominal"]
    while isinstance(central, Varied):
        central = central._members["nominal"]
    return varied_class_for(type(central))(members, tags=tags, context=context)


def expand(fn: Callable[..., Any], args: Sequence[object], kwargs: Mapping[str, object]) -> Any:
    """Call `fn` once per label of the §2.4 union with every operand narrowed to that label.

    With no container among the operands the call passes straight through — an unvaried expression
    must record exactly as it does today, never as a one-member container.

    A per-label answer of `NotImplemented` propagates unchanged, so Python's reflected-operand
    protocol still works through the container.
    """
    operands = (*args, *kwargs.values())
    if not containers_in(*operands):
        return fn(*args, **kwargs)
    results: dict[str, Member] = {}
    for label in union_labels(*operands):
        answer = fn(
            *(narrow(arg, label) for arg in args),
            **{key: narrow(value, label) for key, value in kwargs.items()},
        )
        if answer is NotImplemented:
            return NotImplemented
        results[label] = answer
    return rebuild(results, tags=union_tags(*operands), context=most_derived_context(*operands))


def expand_tuple(fn: Callable[..., Any], args: Sequence[object], kwargs: Mapping[str, object]) -> Any:
    """The *tuple-returning* shape: one `Varied` per tuple position (§2.3c)."""
    operands = (*args, *kwargs.values())
    if not containers_in(*operands):
        return fn(*args, **kwargs)
    labels = union_labels(*operands)
    per_label = {
        label: fn(
            *(narrow(arg, label) for arg in args),
            **{key: narrow(value, label) for key, value in kwargs.items()},
        )
        for label in labels
    }
    width = len(per_label["nominal"])
    tags, context = union_tags(*operands), most_derived_context(*operands)
    return tuple(
        rebuild({label: per_label[label][i] for label in labels}, tags=tags, context=context)
        for i in range(width)
    )


# ---- §2.3d's two mapping dispositions --------------------------------------------------------
def broadcasting(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Record the verb per universe, answering with a `Varied` over the §2.4 union."""

    @functools.wraps(fn)
    def verb(*args: Any, **kwargs: Any) -> Any:
        return expand(fn, args, kwargs)

    return verb


def expanding(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Answer with `{label: <the verb's own return type>}` — `project`'s `Projection` and
    `project_buffers`' `BufferProjection` are distinct types, so neither is wrapped in a
    container. `apply` and `read_columns` expand too but each has its own answer shape."""

    @functools.wraps(fn)
    def verb(*args: Any, **kwargs: Any) -> Any:
        operands = (*args, *kwargs.values())
        if not containers_in(*operands):
            return fn(*args, **kwargs)
        return {
            label: fn(
                *(narrow(arg, label) for arg in args),
                **{key: narrow(value, label) for key, value in kwargs.items()},
            )
            for label in union_labels(*operands)
        }

    return verb


# ---- the §2.3a public-surface parity ------------------------------------------------------
#: §2.3a: every public `Array` method and dunder carries a per-class disposition. *broadcast* maps
#: the call over the universes; *refusing* is §5.4's boundary refusal. Container mechanics
#: (`__init__`/`__repr__`/`__iter__`) are the container's own and carry no disposition.
_BROADCAST_SURFACE = (
    "__abs__", "__add__", "__and__", "__array_ufunc__", "__eq__", "__floordiv__", "__ge__",
    "__getattr__", "__getitem__", "__gt__", "__invert__", "__le__", "__lshift__", "__lt__",
    "__mod__", "__mul__", "__ne__", "__neg__", "__or__", "__pos__", "__pow__", "__radd__",
    "__rand__", "__rfloordiv__", "__rlshift__", "__rmod__", "__rmul__", "__ror__", "__rpow__",
    "__rrshift__", "__rshift__", "__rsub__", "__rtruediv__", "__rxor__", "__sub__",
    "__truediv__", "__xor__", "filter", "map", "reduce",
)  # fmt: skip
SURFACE_DISPOSITIONS: dict[str, str] = dict.fromkeys(_BROADCAST_SURFACE, "broadcast") | {
    "repartition": "refusing"
}


def _broadcast_method(name: str) -> Callable[..., Any]:
    def method(self: Varied, *args: object, **kwargs: object) -> Any:
        return expand(
            lambda member, *rest, **rest_kw: getattr(member, name)(*rest, **rest_kw),
            (self, *args),
            kwargs,
        )

    method.__name__ = name
    method.__qualname__ = f"Varied.{name}"
    return method


def _refusing_method(name: str) -> Callable[..., Any]:
    def method(self: Varied, *args: object, **kwargs: object) -> Any:
        raise boundary_refusal(name, self, *args, *kwargs.values())

    method.__name__ = name
    method.__qualname__ = f"Varied.{name}"
    return method


def install_surface(varied_cls: type[Varied], dispositions: Mapping[str, str]) -> None:
    """Give `varied_cls` a real attribute for every disposed name — §2.3a resolves ON THE CLASS,
    which the instance `__getattr__` fallback cannot satisfy."""
    for name, kind in dispositions.items():
        if name in varied_cls.__dict__:  # a hand-written override wins (`__getattr__`, properties)
            continue
        setattr(varied_cls, name, _broadcast_method(name) if kind == "broadcast" else _refusing_method(name))


install_surface(Varied, SURFACE_DISPOSITIONS)

#: idiom `Array` class -> its `Varied` subclass (`Session._array_cls`'s seam, mirrored)
_VARIED_CLASSES: dict[type, type[Varied]] = {}


def register_varied(array_cls: type[Array], varied_cls: type[Varied]) -> None:
    _VARIED_CLASSES[array_cls] = varied_cls


def varied_class_for(array_cls: type) -> type[Varied]:
    for cls in array_cls.__mro__:
        registered = _VARIED_CLASSES.get(cls)
        if registered is not None:
            return registered
    return Varied


# ---- §2.3d's two refusal contracts ---------------------------------------------------------
def boundary_refusal(verb: str, *values: object) -> GraphedError:
    """§5.4's message shape, shared by every spelling of the boundary refusal (the module verbs, the
    `Varied` method surface, and gak's own dispatch).

    Worded over what the site actually knows: there is no boundary NODE at an operand check, so it
    names the refusing VERB and EVERY label the offending containers carry — `"nominal"` included,
    since a boundary refuses the whole container, not its non-nominal half.
    """
    return GraphedError(
        f"{verb} is a boundary operation and does not accept a Varied carrying labels "
        f"[{', '.join(union_labels(*values))}]; apply it to one universe at a time with "
        "graphed.universe(v, label)"
    )


def refuse_boundary(verb: str, *values: object) -> None:
    """Contract one: boundary and plan verbs (§5.4). They move or key rows, which has no
    per-universe meaning, so they refuse rather than silently compiling one universe."""
    if containers_in(*values):
        raise boundary_refusal(verb, *values)


def refuse_container(verb: str, *values: object) -> None:
    """Contract two: the compile/aggregate verbs. They consume `arr.node_id`/`arr.session`
    directly, and §2.2's reserved-name rule makes that a clean seam — so the refusal points at the
    extraction verb instead of dying on an AttributeError."""
    if containers_in(*values):
        raise GraphedError(
            f"{verb} does not accept a Varied output; pass one universe with "
            "graphed.universe(v, label), or build the varied plan through the histogram group API"
        )
