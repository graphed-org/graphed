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

from .array import Array
from .errors import GraphedError
from .varied import Varied, member_of, rebuild

if TYPE_CHECKING:
    from .context import EventContext

#: §2.2's input shapes. Deliberately NOT `Array`: a plain array carries no universes, and the
#: §2.3d gate reads parameter annotations, so naming `Array` here would ask for a disposition class
#: that does not exist for an accessor.
Introspectable = Any


def _is_context(value: object) -> TypeGuard[EventContext]:
    from .context import EventContext  # noqa: PLC0415  (import cycle: context reads these verbs)

    return isinstance(value, EventContext)


def labels(x: Introspectable) -> tuple[str, ...]:
    """The variation labels `x` carries, `"nominal"` first then insertion order (§2.2)."""
    if isinstance(x, Varied):
        return tuple(x._members)
    if _is_context(x):
        return x._context_labels()
    if isinstance(x, Array):
        raise GraphedError(
            "a plain Array carries no variations; graphed.labels takes a Varied, an event context, a result mapping or a histogram"
        )
    if isinstance(x, Mapping):
        return ("nominal", *(label for label in x if label != "nominal"))
    if hasattr(x, "axes"):  # a bare unvaried histogram reads as the single label "nominal"
        return ("nominal",)
    raise GraphedError(f"graphed.labels does not know how to read {type(x).__name__}")


def universe(x: Introspectable, label: str) -> Any:
    """`x`'s universe for `label` — a KeyError listing the valid labels when it has none (§2.5)."""
    if isinstance(x, Varied):
        return x._universe(label)
    if _is_context(x):
        return x._project(label)
    if isinstance(x, Array):
        raise GraphedError("a plain Array carries no variations; read one from a Varied instead")
    if isinstance(x, Mapping):
        if label not in x:
            raise KeyError(f"unknown variation label {label!r}; this result carries {list(x)}")
        return x[label]
    if hasattr(x, "axes"):
        if label != "nominal":
            raise KeyError(f"unknown variation label {label!r}; this histogram carries ['nominal']")
        return x
    raise GraphedError(f"graphed.universe does not know how to read {type(x).__name__}")


def nominal(x: Introspectable) -> Any:
    """The central universe — `graphed.universe(x, "nominal")` for every shape (§2.2)."""
    return universe(x, "nominal")


def context_of(value: Array | Varied) -> Any:
    """The §2.3e context handle this value was read through, `None` when it is context-free.

    On a `Varied` it answers with the CONTAINER's handle — the most-derived member handle §2.1
    binds — which may belong to a non-nominal member.
    """
    return getattr(value, "_context", None)


def weight(ctx: Any) -> Varied | None:
    """A context's ambient event weight as a `Varied`, `None` when nothing is registered (§9.1).

    Read-only: it returns the registry's current container, it never mutates.
    """
    if not _is_context(ctx):
        raise GraphedError("graphed.weight reads an event context's ambient weight registry")
    return ctx._ambient_weight()


def unify_contexts(*handles: Any) -> Any:
    """§6.1d(A): the most-derived handle when the non-`None` arguments lie on ONE ancestry chain.

    `None` when every argument is context-free; context-free arguments beside contexted ones are
    ignored (the adopt rule); divergent branches raise the §2.3e error naming both contexts.
    """
    present = [handle for handle in handles if handle is not None]
    if not present:
        return None
    best = present[0]
    for handle in present[1:]:
        if handle is best:
            continue
        if best._is_ancestor_of(handle):
            best = handle
        elif not handle._is_ancestor_of(best):
            raise GraphedError(
                f"variation contexts {best!r} and {handle!r} are on divergent branches; one "
                "operation cannot combine values selected differently — derive both from one context"
            )
    return best


def reindex_to(value: Array | Varied, ctx: Any) -> Any:
    """§6.1d(B): `value` re-expressed in `ctx`'s row space, label-aligned per §2.4.

    Identity when `value` already carries `ctx`'s handle or carries none; a `GraphedError` when
    `value`'s handle is a DESCENDANT of `ctx` (a mask has no inverse) or divergent from it.
    """
    handle = context_of(value)
    if handle is None or handle is ctx:
        return value
    if ctx is None:
        raise GraphedError(f"cannot re-index a value read through {handle!r} to a context-free target")
    if not handle._is_ancestor_of(ctx):
        if ctx._is_ancestor_of(handle):
            raise GraphedError(
                f"{handle!r} is a descendant of {ctx!r}: a selection-scoped value has no way back "
                "to its parent's row space (a mask has no inverse) — read the value at "
                f"{ctx!r} instead"
            )
        raise GraphedError(
            f"variation contexts {handle!r} and {ctx!r} are on divergent branches; no lineage path "
            "re-indexes one to the other"
        )
    for kind, payload in ctx._links_below(handle):
        value = _follow(value, kind, payload)
    return with_context(value, ctx)


def _follow(value: Any, kind: str, payload: Any) -> Any:
    from .varied import expand  # noqa: PLC0415  (import cycle)

    if kind == "mask":  # link kind (1): each label's member by THAT label's mask
        return expand(lambda item, mask: item[mask], (value, payload), {})
    if kind == "project":  # link kind (3): project, and RESET the accumulated label set
        return member_of(value, payload)
    return value  # link kind (2): a `vary` link is the identity — only registrations differ


def with_context(value: Any, ctx: Any) -> Any:
    """Stamp `ctx`'s handle on a value (§2.3e's ORIGINATION rule), leaving node identity alone."""
    if isinstance(value, Varied):
        return rebuild(
            {label: with_context(member, ctx) for label, member in value._members.items()},
            tags=value._tags,
            context=ctx,
        )
    if isinstance(value, Array):
        stamped = type(value)(value._session, value._node_id)
        stamped._context = ctx
        stamped._labels = value._labels
        return stamped
    return value


def broadcast_like(value: Array | Varied, factor: Any) -> Any:
    """§6.1d's neutral broadcast seam: `factor` broadcast to `value`'s structure.

    Dispatched to the backend idiom — the awkward backend records `ak.broadcast_arrays`, while a
    rectilinear idiom (numpy) needs nothing, so a backend that supplies no implementation gets the
    bound NO-OP and a genuine shape mismatch surfaces as its own execution-time error.
    """
    from .varied import expand  # noqa: PLC0415  (import cycle)

    def one(item: Any, other: Any) -> Any:
        session = getattr(item, "session", None)
        seam = getattr(session.backend, "broadcast_like", None) if session is not None else None
        return other if seam is None else seam(item, other)

    return expand(one, (value, factor), {})
