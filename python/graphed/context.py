"""The §2.6 event context: systematics attach to `events`, functionally.

A frontend wrapper over the root event record carrying (a) the collections and (b) an ambient
event-weight registry. Pure sugar over §§2.1-2.5 — no IR change, §3.1 intact.

Three properties do the work. Contexts are IMMUTABLE, so `graphed.vary` returns a new one and a
fill from a pre-`vary` context is unaffected by later calls by construction. They RESERVE NO
NAMES, so a tree branch called `weights` or `vary` stays reachable and every graphed operation on
a context is a module function. And a read performed THROUGH a context yields THAT context's row
space, which is what makes `sel.Jet` mean `events.Jet` re-indexed by `sel`'s derivation mask,
label-aligned when that mask is `Varied`.

The neutral mechanism lives here; the nanoevents-flavored constructor is awkward-idiom and lives
in `graphed.awkward.gnano` (the §2.1 factorization rule).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from . import accessors
from ._tags import canonical_tag
from .array import Array
from .by_label import cone
from .errors import GraphedError
from .provenance import capture
from .varied import Varied, expand, labels_of, member_of, rebuild
from .vary import central_universe, check_members, gather_members, register

#: contexts are compared by IDENTITY (§2.6b), so a divergence error must name two distinguishable
#: objects; the serial plus the user line where the context was built is that name
_SERIAL = itertools.count()

#: ("mask", mask) derives rows, ("vary", None) only registrations, ("project", label) narrows to
#: one universe — §6.1d's three link kinds, in one place
Link = tuple[str, Any]


class EventContext:
    """An immutable event context (§2.6). Built by an idiom constructor, never directly."""

    __slots__ = (
        "_collections", "_derived", "_is_data", "_link", "_parent",
        "_projected", "_provenance", "_record", "_serial", "_session", "_weight",
    )  # fmt: skip

    def __init__(
        self,
        session: Any,
        record: Array | Varied,
        *,
        is_data: bool = False,
        parent: EventContext | None = None,
        link: Link | None = None,
        collections: Mapping[str, Any] | None = None,
        weight: Varied | None = None,
    ) -> None:
        self._session = session
        self._record = record
        self._is_data = is_data
        self._parent = parent
        self._link = link
        self._collections: dict[str, Any] = dict(collections or {})
        self._weight = weight
        self._serial = next(_SERIAL)
        self._provenance = capture()
        self._derived: dict[tuple[tuple[str, int], ...], EventContext] = {}
        self._projected: dict[str, EventContext] = {}

    def __repr__(self) -> str:
        return f"EventContext(#{self._serial} from {self._provenance})"

    # ---- tree content (§2.6a: the context reserves NO names) --------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._read(name)

    def __getitem__(self, key: object) -> Any:
        if isinstance(key, str):
            return self._read(key)
        if isinstance(key, list) and key and all(isinstance(field, str) for field in key):
            return self._stamp(expand(lambda record: record[key], (self._record,), {}))
        if isinstance(key, Array | Varied):
            return self._derive(key)
        raise GraphedError(
            f"an event context takes a field name, a list of field names, or an Array/Varied mask "
            f"as its subscript, not {key!r}"
        )

    def __iter__(self) -> Any:
        raise TypeError("an event context is not iterable; subscript it with a mask or a field name")

    def _read(self, name: str) -> Any:
        value = self._collections.get(name)
        if value is None:
            value = expand(lambda record: getattr(record, name), (self._record,), {})
        return self._stamp(value)

    def _stamp(self, value: Any) -> Any:
        """§2.3e's ORIGINATION rule: everything a context produces carries THAT context's handle,
        overriding whatever the input merge would yield (a derived context's reads go through the
        same root wrapper, so the merge alone would answer with the parent's handle)."""
        return accessors.with_context(value, self)

    # ---- lineage (§2.6b: variation history is object lineage) -------------------
    def _is_ancestor_of(self, other: EventContext) -> bool:
        node: EventContext | None = other
        while node is not None:
            if node is self:
                return True
            node = node._parent
        return False

    def _links_below(self, ancestor: EventContext) -> tuple[Link, ...]:
        """The links from just below `ancestor` down to `self`, parent-to-child (§6.1d)."""
        links: list[Link] = []
        node: EventContext | None = self
        while node is not None and node is not ancestor:
            if node._link is not None:
                links.append(node._link)
            node = node._parent
        return tuple(reversed(links))

    def _selection(self) -> Any:
        """§9.1's selection: the mask that derived this context, skipping `vary` IDENTITY links
        and answering as of the first non-identity link (`None` at a root or across a projection,
        which is one label's unvaried member and contributes no labels)."""
        node: EventContext | None = self
        while node is not None and node._link is not None:
            kind, payload = node._link
            if kind == "mask":
                return payload
            if kind == "project":
                return None
            node = node._parent
        return None

    def _ambient_weight(self) -> Varied | None:
        return self._weight

    def _context_labels(self) -> tuple[str, ...]:
        """§2.2's §2.4-ordered union: (a) the ambient registry's labels, (b) the labels of the
        `Varied` collections this context CARRIES, (c) the labels of its selection."""
        out: dict[str, None] = {"nominal": None}
        for source in (self._weight, *self._collections.values(), self._selection()):
            for label in labels_of(source):
                out.setdefault(label, None)
        return tuple(out)

    # ---- derivation (§2.6c: scoping is lineage) ---------------------------------
    def _derive(self, mask: Array | Varied) -> EventContext:
        """`ctx[mask]`. PURE DERIVATIONS ARE CANONICAL — the same mask answers with the same
        object, memoised here, or two reads of one universe would falsely trip §2.3e's divergence
        rule."""
        key = _mask_key(mask)
        memo = self._derived.get(key)
        if memo is not None:
            return memo
        child = EventContext(
            self._session,
            expand(lambda record, on: record[on], (self._record, mask), {}),
            is_data=self._is_data,
            parent=self,
            link=("mask", mask),
        )
        # every member re-indexed by THAT label's own mask, label-aligned per §2.4
        child._collections = {
            name: child._stamp(expand(lambda value, on: value[on], (collection, mask), {}))
            for name, collection in self._collections.items()
        }
        if self._weight is not None:
            child._weight = child._stamp(expand(lambda value, on: value[on], (self._weight, mask), {}))
        child._record = child._stamp(child._record)
        self._derived[key] = child
        return child

    def _project(self, label: str) -> EventContext:
        """`graphed.universe(ctx, L)` / `graphed.nominal(ctx)`: a CHILD context carrying that
        label's collections and ambient weight (§2.2)."""
        if label not in self._context_labels():
            raise KeyError(
                f"unknown variation label {label!r}; this context carries {list(self._context_labels())}"
            )
        memo = self._projected.get(label)
        if memo is not None:
            return memo
        child = EventContext(
            self._session,
            member_of(self._record, label),
            is_data=self._is_data,
            parent=self,
            link=("project", label),
        )
        child._collections = {
            name: child._stamp(member_of(collection, label)) for name, collection in self._collections.items()
        }
        if self._weight is not None:
            child._weight = child._stamp(member_of(self._weight, label))
        child._record = child._stamp(child._record)
        self._projected[label] = child
        return child


def _mask_key(mask: Array | Varied) -> tuple[tuple[str, int], ...]:
    """A derivation's identity: its per-label node ids. A REBUILT mask interns to the same ids,
    which is the binding condition for `ctx[rebuilt] is ctx[mask]`."""
    if isinstance(mask, Varied):
        return tuple((label, member.node_id) for label, member in mask._members.items())
    return (("nominal", mask.node_id),)


# ---- the two context overloads of `graphed.vary` -------------------------------------------
def vary_context(
    ctx: EventContext,
    name: str,
    nominal: object,
    is_weight: bool,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    tags: Mapping[str, Any],
) -> EventContext:
    if ctx._is_data:
        raise GraphedError(
            f"variation {name!r} cannot be registered on a data context: data fills nominal-only, "
            "and accepting a registration whose labels the fill then drops would be a silent drop"
        )
    if is_weight:
        return _vary_weight(ctx, name, nominal, variations, collections, tags)
    return _vary_shift(ctx, name, nominal, variations, collections, tags)


def _child_of(ctx: EventContext) -> EventContext:
    """A `vary` link: the row space is unchanged, only registrations differ (§6.1d kind (2))."""
    child = EventContext(
        ctx._session,
        ctx._record,
        is_data=ctx._is_data,
        parent=ctx,
        link=("vary", None),
        collections=ctx._collections,
        weight=ctx._weight,
    )
    child._record = child._stamp(child._record)
    return child


def _vary_weight(
    ctx: EventContext,
    name: str,
    central: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    tags: Mapping[str, Any],
) -> EventContext:
    """Overload (b): register a per-event weight factor into the returned context's ambient
    weight, composed label-aligned per §2.4 with whatever is already registered."""
    if collections is not None:
        raise GraphedError("collections= belongs to the shift form; a weight form takes tags")
    if central is None:
        raise GraphedError(
            f"the weight form of graphed.vary({name!r}) needs the central per-event factor as its "
            "third positional argument"
        )
    old = ctx._weight
    inherited = old._tags.get(name, ()) if old is not None else ()
    members = gather_members(name, tags, variations, inherited)
    # §1.1's within-the-container clause, keyed by family NAME over the three carriers
    # `_context_labels` reads. `_members` alone is the §2.4 union and cannot say which family a
    # label came from; the same `name` is the correlated case (one knob, §2.1) and is admitted —
    # `check_family` refuses a repeated tag within it — while any OTHER family already spelling
    # this label would make one universe differ from nominal in two knobs.
    registered = {
        f"{n}_{t}"
        for source in (old, *ctx._collections.values(), ctx._selection())
        for n, ts in (getattr(source, "_tags", {}) or {}).items()
        if n != name
        for t in ts
    }
    for label in members:
        if label in registered:
            raise GraphedError(f"variation label {label!r} is already carried by this container")
    factors = {"nominal": central, **members}
    check_members(factors)
    # §2.1(b)'s ROW-SPACE rule: an ancestor-handled factor is re-indexed across the intervening
    # links; a descendant or divergent one is a construction-time error naming the direction.
    factors = {label: accessors.reindex_to(factor, ctx) for label, factor in factors.items()}
    factor = rebuild(factors, tags={name: inherited + _tags_of(name, factors)}, context=ctx)
    # §2.5's shift-after-weight operand one: this factor's OWN member node ids, by value.
    ctx._session._weight_factors.append((name, _member_nodes(factor)))

    child = _child_of(ctx)
    ambient: dict[str, Any] = {}
    for label in _union(ctx._context_labels(), tuple(factors)):
        applied = _two_level(factor, label)
        ambient[label] = applied if old is None else _two_level(old, label) * applied
    tag_map = dict(old._tags) if old is not None else {}
    tag_map[name] = inherited + _tags_of(name, members)
    child._weight = child._stamp(register(rebuild(ambient, tags=tag_map, context=child)))
    return child


def _vary_shift(
    ctx: EventContext,
    name: str,
    nominal: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    tags: Mapping[str, Any],
) -> EventContext:
    """Overload (c): replace each named collection with a `Varied` over one shared tag set."""
    if nominal is not None:
        raise GraphedError(
            "nominal= has no meaning in the shift form — the collections' central members come "
            "from the target context; name the collections with collections={Name: {tag: record}}"
        )
    if variations is not None:
        raise GraphedError(
            "variations= is not accepted in the shift form; its tags are the INNER keys of the "
            "collection mappings, so pass collections={Name: {tag: record}}"
        )
    mapping: dict[str, Mapping[Any, Any]] = dict(tags)
    for collection_name, inner in (collections or {}).items():
        if collection_name in mapping:
            raise GraphedError(f"collection {collection_name!r} was named twice")
        mapping[collection_name] = inner
    if not mapping:
        raise GraphedError(f"the shift form of graphed.vary({name!r}) needs at least one collection")
    _check_lockstep(name, mapping)

    child = _child_of(ctx)
    replaced = dict(ctx._collections)
    for collection_name, inner in mapping.items():
        current = ctx._read(collection_name)
        inherited = current._tags.get(name, ()) if isinstance(current, Varied) else ()
        members = gather_members(name, inner, None, inherited)
        existing = dict(current._members) if isinstance(current, Varied) else {"nominal": current}
        resolved = {label: central_universe(member) for label, member in members.items()}
        check_members({**existing, **resolved})
        resolved = {label: accessors.reindex_to(member, ctx) for label, member in resolved.items()}
        for label in resolved:
            if label in existing:
                raise GraphedError(f"variation label {label!r} already varies {collection_name!r}")
        tag_map = dict(current._tags) if isinstance(current, Varied) else {}
        tag_map[name] = inherited + _tags_of(name, resolved)
        replaced[collection_name] = child._stamp(
            register(rebuild({**existing, **resolved}, tags=tag_map, context=ctx))
        )
        _report_shift_after_weight(ctx, collection_name, existing)
    child._collections = replaced
    return child


def _member_nodes(value: Any) -> tuple[int, ...]:
    """A container's member node ids, resolving §2.2's one legal level of nesting."""
    if not isinstance(value, Varied):
        return (value.node_id,)
    return tuple(nid for member in value._members.values() for nid in _member_nodes(member))


def _report_shift_after_weight(ctx: EventContext, collection: str, pre_shift: Mapping[str, Any]) -> None:
    """§2.5/§2.1: a weight factor registered BEFORE the collection it reads is varied fills every
    shift universe with its PRE-shift value, and the registry is not re-derived, so it is
    unfixable after the fact. Report each such family paired with the collection it reads.

    Diagnostic, not an error: a weight that legitimately does not track the shift is a valid
    program — which is why the walk is per (family, collection) rather than a membership test.
    """
    session = ctx._session
    if not session._weight_factors:
        return
    targets = {member.node_id for member in pre_shift.values()}
    registry = session._shift_after_weight
    for family, nodes in session._weight_factors:
        if any(targets & cone(session, nid) for nid in nodes):
            key = (family, collection)
            # by value, with the factor's own ids: the shipping site filters on them (§2.5's
            # report is about one compiled program, the registry is about the whole Session)
            registry[key] = registry.get(key, frozenset()) | frozenset(nodes)


def _check_lockstep(name: str, mapping: Mapping[str, Mapping[Any, Any]]) -> None:
    """§2.6a: all collections in one call MUST share one tag set (the lockstep Jet+MET form)."""
    sets = {}
    for collection_name, inner in mapping.items():
        if not isinstance(inner, Mapping):
            raise GraphedError(
                f"collection {collection_name!r} needs a {{tag: record}} mapping, got {type(inner).__name__}"
            )
        sets[collection_name] = frozenset(canonical_tag(tag) for tag in inner)
    if len({frozenset(tags) for tags in sets.values()}) > 1:
        raise GraphedError(
            f"variation {name!r} moves its collections out of lockstep: "
            f"{ {key: sorted(value) for key, value in sets.items()} } — one call's collections must "
            "share one tag set"
        )


def _two_level(container: Any, label: str) -> Any:
    """§2.1's `factor[L]`: the container's member for L, then — when that member is ITSELF a
    `Varied` (a registered factor computed on shifted objects) — its own member for L. The
    composed ambient weight is therefore always FLAT."""
    return member_of(member_of(container, label), label)


def _union(*groups: Sequence[str]) -> tuple[str, ...]:
    out: dict[str, None] = {"nominal": None}
    for group in groups:
        for label in group:
            out.setdefault(label, None)
    return tuple(out)


def _tags_of(name: str, members: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(label[len(name) + 1 :] for label in members if label != "nominal")
