"""What an event context is to the systematics layers: its Session, its lineage, its collections.

An event context reserves every public name for tree fields (§2.6a), so its systematics state
cannot grow public members. `ContextNode` is the base class that carries it, and the functions
here are its interface: every other module reads a context's lineage through them.

Variation history is object lineage (§2.6b): a context is derived from its parent by one link —
`("mask", mask)` derives rows, `("vary", None)` changes only registrations, `("project", label)`
narrows to one universe (§6.1d).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, TypeVar

from ..errors import GraphedError

#: §6.1d's three link kinds: ("mask", mask), ("vary", None), ("project", label)
Link = tuple[str, Any]

_Node = TypeVar("_Node", bound="ContextNode")


class ContextNode:
    """An event context's Session, parent, link and collections."""

    __slots__ = ("_collections", "_link", "_parent", "_session")

    _parent: Self | None

    def __init__(
        self,
        session: Any,
        parent: Self | None,
        link: Link | None,
        collections: Mapping[str, Any] | None,
    ) -> None:
        """Derive a context from ``parent`` by ``link``, carrying ``collections``."""
        self._session = session
        self._parent = parent
        self._link = link
        self._collections: dict[str, Any] = dict(collections or {})


def context_session(ctx: ContextNode) -> Any:
    """Return the Session `ctx` records into."""
    return ctx._session


def parent_of(ctx: _Node) -> _Node | None:
    """Return the context `ctx` was derived from, `None` at a root."""
    return ctx._parent


def link_of(ctx: ContextNode) -> Link | None:
    """Return the link that derived `ctx` from its parent, `None` at a root."""
    return ctx._link


def collections_of(ctx: ContextNode) -> Mapping[str, Any]:
    """Return the collections `ctx` carries, by name."""
    return ctx._collections


def is_ancestor(ancestor: ContextNode, ctx: ContextNode) -> bool:
    """Whether `ancestor` is `ctx` or lies on its parent chain."""
    node: ContextNode | None = ctx
    while node is not None:
        if node is ancestor:
            return True
        node = node._parent
    return False


def links_below(ctx: ContextNode, ancestor: ContextNode) -> tuple[Link, ...]:
    """Return the links from just below `ancestor` down to `ctx`, parent-to-child (§6.1d)."""
    links: list[Link] = []
    node: ContextNode | None = ctx
    while node is not None and node is not ancestor:
        if node._link is not None:
            links.append(node._link)
        node = node._parent
    return tuple(reversed(links))


def selection(ctx: ContextNode) -> Any:
    """Return the mask that derived `ctx`'s row space.

    It skips `vary` identity links, answering as of the first non-identity link: `None` at a root
    or across a projection, which is one label's unvaried member and contributes no labels. Not
    §9.1's `graphed.selection`, whose projection case answers with the parent's selection.
    """
    node: ContextNode | None = ctx
    while node is not None and node._link is not None:
        kind, payload = node._link
        if kind == "mask":
            return payload
        if kind == "project":
            return None
        node = node._parent
    return None


def row_space_above(handle: ContextNode, ctx: ContextNode) -> ContextNode | None:
    """Return the context `handle`'s row space belongs to as seen from `ctx`.

    `handle` itself is not one of `ctx`'s ancestors, but a run of `vary` links above it may end at
    one (or at `ctx`), and such a run changes no rows. `None` when the two are on genuinely
    different row spaces.
    """
    node = handle
    while node._link is not None and node._link[0] == "vary" and node._parent is not None:
        node = node._parent
        if node is ctx or is_ancestor(node, ctx):
            return node
    return None


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
        if is_ancestor(best, handle):
            best = handle
        elif not is_ancestor(handle, best):
            raise GraphedError(
                f"variation contexts {best!r} and {handle!r} are on divergent branches; one "
                "operation cannot combine values selected differently — derive both from one "
                "context"
            )
    return best
