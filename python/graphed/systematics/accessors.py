"""§9.1's introspection surface: the module verbs that read universes, handles and lineage.

Extraction is functional, never a method or a subscript (§2.2/§2.6a): branch names are
analysis-controlled and open-ended, so any reserved attribute on a `Varied` or an event context
would be a latent collision with real tree content. `graphed.labels`/`universe`/`nominal` take the
same four input shapes — a `Varied`, an event context, a `{label: hist}` result mapping, and a
duck-typed histogram — so introspection reads uniformly wherever a variation can end up.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeGuard

from ..array import Array
from ..context import project
from ..errors import GraphedError
from .ambient import context_labels, weight_of
from .handles import broadcast_like as broadcast_like
from .handles import reindex_to as reindex_to
from .handles import with_context as with_context
from .kinds import Kind
from .lineage import (
    ContextNode,
    ambient_of,
    collections_of,
    context_session,
    link_of,
    parent_of,
)
from .lineage import selection as lineage_selection
from .lineage import unify_contexts as unify_contexts
from .points import Point, render
from .tags import numeric_value
from .varied import (
    Varied,
    member_exact,
    member_of,
    members_of,
    point_registry,
    tags_of,
)
from .varied import context_of as context_of

if TYPE_CHECKING:
    from fractions import Fraction

    from ..context import EventContext

#: §2.2's input shapes. Deliberately NOT `Array`: a plain array carries no universes, and the
#: §2.3d gate reads parameter annotations, so naming `Array` here would ask for a disposition class
#: that does not exist for an accessor.
Introspectable = Any


def _is_context(value: object) -> TypeGuard[EventContext]:
    return isinstance(value, ContextNode)


def _variation_axis_index(hist: Any) -> int | None:
    """§6.2(i-bis): the position of the axis-mode variation axis, `None` when there is none.

    Recognised by the axis' `name` — a `hist` axis takes it by keyword, while
    `bh.axis.StrCategory(..., name=)` is a `TypeError`, so the frontend (and the fixtures) stamp
    it on the axis' `__dict__`. `hist.axes.name` is
    NOT the oracle: bh maps that attribute over every axis and raises when one lacks it.
    """
    return next(
        (i for i, a in enumerate(hist.axes) if getattr(a, "name", None) == "variation"), None
    )


def labels(x: Introspectable) -> tuple[str, ...]:
    """Return the variation labels `x` carries, `"nominal"` first then insertion order (§2.2)."""
    if isinstance(x, Varied):
        return tuple(members_of(x))
    if _is_context(x):
        return context_labels(x)
    if isinstance(x, Array):
        raise GraphedError(
            "a plain Array carries no variations; graphed.labels takes a Varied, an event "
            "context, a result mapping or a histogram"
        )
    if isinstance(x, Mapping):
        seat = ("nominal",) if "nominal" in x else ()
        return (*seat, *(label for label in x if label != "nominal"))
    if hasattr(x, "axes"):  # a bare histogram: an axis-mode variation axis, else "nominal"-only
        index = _variation_axis_index(x)
        if index is not None:  # §6.2: reorder "nominal"-first over the lexicographic stored order
            stored = tuple(x.axes[index])
            seat = ("nominal",) if "nominal" in stored else ()
            return (*seat, *(bin_ for bin_ in stored if bin_ != "nominal"))
        return ("nominal",)
    raise GraphedError(f"graphed.labels does not know how to read {type(x).__name__}")


def universe(x: Introspectable, label: str) -> Any:
    """`x`'s universe for `label` — a KeyError listing the valid labels when it has none (§2.5)."""
    if isinstance(x, Varied):
        return member_exact(x, label)
    if _is_context(x):
        return project(x, label)
    if isinstance(x, Array):
        raise GraphedError("a plain Array carries no variations; read one from a Varied instead")
    if isinstance(x, Mapping):
        if label not in x:
            raise KeyError(f"unknown variation label {label!r}; this result carries {list(x)}")
        return x[label]
    if hasattr(x, "axes"):
        index = _variation_axis_index(x)
        if index is not None:
            # resolve the bin to an integer position ourselves (§A.4: no boost_histogram import in
            # the frontend). StrCategory.index raises KeyError for an unknown label, exactly as
            # bh.loc did, so the unknown-label path still needs no guard of our own.
            return x[{index: x.axes[index].index(label)}]
        if label != "nominal":
            raise KeyError(
                f"unknown variation label {label!r}; this histogram carries ['nominal']"
            )
        return x
    raise GraphedError(f"graphed.universe does not know how to read {type(x).__name__}")


def nominal(x: Introspectable) -> Any:
    """Return the central universe — `graphed.universe(x, "nominal")` for every shape (§2.2)."""
    return universe(x, "nominal")


def points(x: Introspectable) -> dict[str, dict[str, str]]:
    """`{label: {nuisance: coordinate}}` — the authoritative coordinate view (§4.10).

    Label-sorted, each coordinate map nuisance-sorted, `"nominal"` mapping to `{}`. Coordinates
    render BY VALUE, so the view predicts resolution and deliberately does not round-trip the tag
    spelling the label keeps.

    Defined on the RECORD-TIME shapes alone — a `Varied` and an event context. Points are not
    carried on disk (§5.3) and a label cannot be parsed back into a point (§4.3), so answering
    `{label: {}}` for an executed result would assert that every executed universe is the origin.
    """
    if isinstance(x, Varied):
        registry: Mapping[str, Point] = point_registry(x)
        carried = labels(x)
    elif _is_context(x):
        registry, carried = context_session(x).systematics.points, context_labels(x)
    else:
        raise GraphedError(
            "graphed.points reads a record-time shape — a Varied or an event context — not "
            f"{type(x).__name__}: points are not carried on an executed result, and a label "
            "cannot be parsed back into a point"
        )
    return {label: render(registry.get(label, Point({}))) for label in sorted(carried)}


def selection(ctx: Any) -> Any:
    """§9.1's bridge: the `Varied`/Array mask that derived a context from its parent (§6.4a).

    `None` for a root context. On a universe/nominal-derived context it returns that label's member
    of the argument's own selection — an unvaried `Array` in the grandparent's row space; on a
    `vary`-derived one it skips the identity links and answers with the mask below. The verb that
    makes the m51 skim sink reachable from the §2.6 context idiom. Not `lineage.selection`, which
    answers `None` across a projection.
    """
    if not _is_context(ctx):
        raise GraphedError("graphed.selection reads an event context's derivation mask")
    node: ContextNode | None = ctx
    while node is not None and (link := link_of(node)) is not None:
        kind, payload = link
        if kind == "mask":
            return payload
        parent = parent_of(node)
        if kind == "project":
            parent_selection = lineage_selection(parent) if parent is not None else None
            return None if parent_selection is None else member_of(parent_selection, payload)
        node = parent  # `vary` identity link: skip
    return None


def weight(ctx: Any) -> Varied | Array | None:
    """Return a context's ambient event weight as a `Varied`.

    `None` when nothing is registered (§9.1); a projected context's
    (`graphed.universe(ctx, label)`) is that one universe's member.

    Read-only: it returns the registry's current container, it never mutates.
    """
    if not _is_context(ctx):
        raise GraphedError("graphed.weight reads an event context's ambient weight registry")
    return weight_of(ctx)


def variations(ctx: Any) -> dict[str, dict[str, tuple[Kind, Fraction | None]]]:
    """Return a context's registered variations as `{name: {tag: (kind, value | None)}}` (§9.1).

    The kind is a `graphed.Kind` PER (name, tag): `Kind.WEIGHT` for a §2.1 overload-(b)
    registration, `Kind.SHIFT` for an overload-(c) one, and their union when the tag was registered
    both ways — §4.8's mechanism for "these two registrations are the same fit parameter". The
    weight side is read from the lineage's registration record, never from the ambient container's
    tag map, which a row-space change widens with the shifts the mask carries. The value is the
    tag's parsed numeric magnitude — the ordering handle §6.2's lexicographic axis cannot give —
    and `None` for a non-numeric tag.
    """
    if not _is_context(ctx):
        raise GraphedError("graphed.variations reads an event context's registered variations")
    kinds: dict[str, dict[str, Kind]] = {}
    for name, tags in ambient_of(ctx).weight_tags.items():
        for tag in tags:
            kinds.setdefault(name, {})[tag] = kinds.get(name, {}).get(tag, Kind(0)) | Kind.WEIGHT
    for collection in collections_of(ctx).values():
        for name, tags in tags_of(collection).items():
            for tag in tags:
                kinds.setdefault(name, {})[tag] = (
                    kinds.get(name, {}).get(tag, Kind(0)) | Kind.SHIFT
                )
    return {
        name: {tag: (kind, numeric_value(tag)) for tag, kind in tags.items()}
        for name, tags in kinds.items()
    }
