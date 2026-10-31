"""A value's handle (§2.3e/§6.1d): stamping it, re-indexing a value to a context, broadcasting.

A handle is the event context a value was read through. These are the operations that move a value
between handles; the readers of a handle are `graphed.systematics.accessors`.
"""

from __future__ import annotations

from typing import Any

from ..array import Array, attach, labels_carried
from ..errors import GraphedError
from .lineage import is_ancestor, links_below, row_space_above
from .varied import Varied, context_of, expand, member_of, members_of, rebuild, tags_of


def reindex_to(value: Array | Varied, ctx: Any) -> Any:
    """§6.1d(B): `value` re-expressed in `ctx`'s row space, label-aligned per §2.4.

    Identity when `value` already carries `ctx`'s handle or carries none; a `GraphedError` when
    `value`'s handle is a DESCENDANT of `ctx` (a mask has no inverse) or divergent from it.

    A `vary` link changes only registrations, so a handle standing below a RUN of them stands in
    the row space above that run: the walk starts from there, which is what re-indexes a value read
    on a branch that differs from `ctx` in registrations alone.
    """
    handle = context_of(value)
    if handle is None or handle is ctx:
        return value
    if ctx is None:
        raise GraphedError(
            f"cannot re-index a value read through {handle!r} to a context-free target"
        )
    if not is_ancestor(handle, ctx):
        above = row_space_above(handle, ctx)
        if above is not None:
            handle = above
        elif is_ancestor(ctx, handle):
            raise GraphedError(
                f"{handle!r} is a descendant of {ctx!r}: a selection-scoped value has no way back "
                "to its parent's row space (a mask has no inverse) — read the value at "
                f"{ctx!r} instead"
            )
        else:
            raise GraphedError(
                f"variation contexts {handle!r} and {ctx!r} are on divergent branches; no lineage "
                "path re-indexes one to the other"
            )
    for kind, payload in links_below(ctx, handle):
        value = follow(value, kind, payload)
    return with_context(value, ctx)


def follow(value: Any, kind: str, payload: Any) -> Any:
    """Return `value` carried across one lineage link (§6.1d), parent-to-child."""
    if kind == "mask":  # link kind (1): each label's member by THAT label's mask
        return expand(lambda item, mask: item[mask], (value, payload), {})
    if kind == "project":  # link kind (3): project, and RESET the accumulated label set
        return member_of(value, payload)
    return value  # link kind (2): a `vary` link is the identity — only registrations differ


def with_context(value: Any, ctx: Any) -> Any:
    """Stamp `ctx`'s handle on a value (§2.3e's ORIGINATION rule), leaving node identity alone."""
    if isinstance(value, Varied):
        return rebuild(
            {label: with_context(member, ctx) for label, member in members_of(value).items()},
            tags=tags_of(value),
            context=ctx,
        )
    if isinstance(value, Array):
        stamped = type(value)(value.session, value.node_id)
        stamped._site = value._site
        return attach(stamped, ctx, labels_carried(value))
    return value


def broadcast_like(value: Array | Varied, factor: Any) -> Any:
    """§6.1d's neutral broadcast seam: `factor` broadcast to `value`'s structure.

    Dispatched to the backend idiom — the awkward backend records `ak.broadcast_arrays`, while a
    rectilinear idiom (numpy) needs nothing, so a backend that supplies no implementation gets the
    bound NO-OP and a genuine shape mismatch surfaces as its own execution-time error.
    """

    def one(item: Any, other: Any) -> Any:
        session = getattr(item, "session", None)
        seam = getattr(session.backend, "broadcast_like", None) if session is not None else None
        return other if seam is None else seam(item, other)

    return expand(one, (value, factor), {})
