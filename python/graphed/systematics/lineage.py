"""What an event context is to the systematics layers: its Session, its lineage, its collections.

An event context reserves every public name for tree fields (§2.6a), so its systematics state
cannot grow public members. `ContextNode` is the base class that carries it, and the functions
here are its interface: every other module reads a context's lineage through them.

Variation history is object lineage (§2.6b): a context is derived from its parent by one link —
`("mask", mask)` derives rows, `("vary", None)` changes only registrations, `("project", label)`
narrows to one universe (§6.1d), and `("option", masker)` masks values without changing rows
(`ak.mask`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Self, TypeVar

from ..array import detached
from ..errors import GraphedError

if TYPE_CHECKING:
    from .ambient import AmbientRecord

#: §6.1d's three link kinds, ("mask", mask), ("vary", None), ("project", label), and ak.mask's
#: ("option", masker)
Link = tuple[str, Any]

#: the links that keep the parent's rows and registrations
ROW_PRESERVING = frozenset({"vary", "option"})

_Node = TypeVar("_Node", bound="ContextNode")


class Followed(MutableMapping[str, Any]):
    """A derived context's collections, each carried across its links when first read.

    A collection no one reads records nothing; the parent's order is kept.
    """

    __slots__ = ("_items",)

    _items: dict[str, tuple[Any, tuple[Callable[[Any], Any], ...]]]

    def __init__(self, parent: Mapping[str, Any], carry: Callable[[Any], Any]) -> None:
        """Hold ``parent``'s collections, each to be read through ``carry``."""
        items = (
            parent._items
            if isinstance(parent, Followed)
            else {n: (v, ()) for n, v in parent.items()}
        )
        self._items = {name: (value, (*steps, carry)) for name, (value, steps) in items.items()}

    def __getitem__(self, name: str) -> Any:
        """Return collection ``name``, carried once."""
        value, steps = self._items[name]
        for carry in steps:
            value = carry(value)
        self._items[name] = (value, ())
        return value

    def get(self, name: str, default: Any = None) -> Any:
        """Return collection ``name``, or ``default`` when none is held by that name."""
        # Mapping.get would read a KeyError raised while carrying the value as a missing name
        return self[name] if name in self._items else default

    def __setitem__(self, name: str, value: Any) -> None:
        """Hold ``value`` as collection ``name``, already carried."""
        self._items[name] = (value, ())

    def __delitem__(self, name: str) -> None:
        """Drop collection ``name``."""
        del self._items[name]

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is held, carrying nothing."""
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        """Iterate the names in the parent's order."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return the number of collections."""
        return len(self._items)


class ContextNode:
    """An event context's Session, parent, link, collections and ambient record.

    The record (`ambient_of`) is set by the constructing subclass, which the record names.
    """

    __slots__ = ("_ambient", "_collections", "_link", "_parent", "_session")

    _parent: Self | None
    _ambient: AmbientRecord

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
        self._collections: MutableMapping[str, Any] = {
            name: detached(value) for name, value in (collections or {}).items()
        }

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """Return this context: contexts compare by identity, so a copy would be a stranger."""
        return self

    __copy__ = __deepcopy__


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


def ambient_of(ctx: ContextNode) -> AmbientRecord:
    """Return `ctx`'s ambient weight record (`graphed.systematics.ambient`)."""
    return ctx._ambient


def same_rows(a: ContextNode, b: ContextNode) -> bool:
    """Whether `a` and `b` hold one row space: one context, or one mask derivation of one.

    An assignment on a parent re-derives `parent[mask]` as a new context over the same rows.
    """
    while a is not b:
        if a._link is None or b._link is None or a._link[0] != "mask" or b._link[0] != "mask":
            return False
        same = a._link[1] is b._link[1] or mask_key(a._link[1]) == mask_key(b._link[1])
        if not same or a._parent is None or b._parent is None:
            return False
        a, b = a._parent, b._parent
    return True


def mask_key(mask: Any) -> tuple[tuple[str, int], ...]:
    """Return a mask derivation's identity: its per-label node ids.

    A REBUILT mask interns to the same ids, which is the binding condition for
    `ctx[rebuilt] is ctx[mask]`.
    """
    members = getattr(mask, "_members", None)  # by duck type: `varied` imports this module
    if members is None:
        return (("nominal", mask.node_id),)
    return tuple((label, member.node_id) for label, member in members.items())


def is_ancestor(ancestor: ContextNode, ctx: ContextNode) -> bool:
    """Whether `ancestor` holds the rows of `ctx` or of a context on its parent chain."""
    node: ContextNode | None = ctx
    while node is not None:
        if same_rows(node, ancestor):
            return True
        node = node._parent
    return False


def links_below(ctx: ContextNode, ancestor: ContextNode) -> tuple[Link, ...]:
    """Return the links from just below `ancestor` down to `ctx`, parent-to-child (§6.1d)."""
    links: list[Link] = []
    node: ContextNode | None = ctx
    while node is not None and not same_rows(node, ancestor):
        if node._link is not None:
            links.append(node._link)
        node = node._parent
    return tuple(reversed(links))


def selection(ctx: ContextNode) -> Any:
    """Return the mask that derived `ctx`'s row space.

    It skips `vary` identity links, answering as of the first non-identity link: `None` at a root
    or across a projection, which is one label's unvaried member and contributes no labels. Not
    §9.1's `selection`, whose projection case answers with the parent's selection.
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

    `handle` itself is not one of `ctx`'s ancestors, but a run of `ROW_PRESERVING` links above it
    may end at one (or at `ctx`), and such a run changes no rows. `None` when the two are on
    genuinely different row spaces.
    """
    node = handle
    while node._link is not None and node._link[0] in ROW_PRESERVING and node._parent is not None:
        node = node._parent
        if node is ctx or is_ancestor(node, ctx):
            return node
    return None


def rows_of(value: Any) -> Any:
    """Return the context whose rows `value` holds: where a row cut put it, else its handle."""
    members = getattr(value, "_members", None)
    if members is not None:
        rows = {id(r): r for r in map(rows_of, members.values())}
        return next(iter(rows.values())) if len(rows) == 1 else None
    return value._context if value._rows is None else value._rows


def merged_rows(*rows: Any) -> Any:
    """Return the most-derived of `rows` on one chain, `None` where they diverge.

    A row cut's context is a guess from the mask's rank, so where it diverges it is dropped.
    """
    try:
        return unify_contexts(*rows)
    except GraphedError:
        return None


def unify_contexts(*handles: Any) -> Any:
    """Return the most-derived handle when the non-`None` arguments lie on ONE ancestry chain.

    `None` when every argument is context-free; context-free arguments beside contexted ones are
    ignored (the adopt rule); divergent branches raise an error naming both contexts.
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
