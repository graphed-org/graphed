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
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from . import accessors
from ._tags import canonical_tag
from .array import Array
from .by_label import cone
from .errors import GraphedError, GraphedTypeError
from .provenance import capture
from .varied import Varied, expand, labels_of, member_of, point_registry, rebuild, session_of
from .vary import AmbientCarrier, check_members, gather_members, record_labels, register, stamp_labels

#: contexts are compared by IDENTITY (§2.6b), so a divergence error must name two distinguishable
#: objects; the serial plus the user line where the context was built is that name
_SERIAL = itertools.count()

#: ("mask", mask) derives rows, ("vary", None) only registrations, ("project", label) narrows to
#: one universe — §6.1d's three link kinds, in one place
Link = tuple[str, Any]


class EventContext:
    """An event context (§2.6). Built by an idiom constructor, never directly.

    Its RECORDED state — collections, selection, the registered weight factors and the label union
    over them — never changes after construction, which is what makes `graphed.vary` return a new
    context and a fill from a pre-`vary` one unaffected. The ambient weight is composed from that
    recorded state on demand, and `_memo` is a pure CACHE of it stamped with the Session's mint
    epoch: a mint landing after a composition makes the next read remake it from the same original
    factors, so what a factor contributes at a label is what the registry says AS OF THE READ.
    """

    __slots__ = (
        "_collections", "_derived", "_factors", "_is_data", "_link", "_memo",
        "_origin", "_parent", "_projected", "_provenance", "_record", "_recorded", "_serial",
        "_session",
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
        # §2.6's ambient weight is a DEFERRED composition (§4): the registered factors in
        # registration order, the label union RECORDED at each registration, and the composed
        # container once something reads it. Composing at every registration is quadratic in the
        # families registered whatever the association, because each one rewrites every label the
        # ambient already carries.
        self._factors: list[Any] = [] if weight is None else [weight]
        self._recorded: tuple[str, ...] = _union(("nominal",), labels_of(weight))
        #: the cache: `(mint epoch, factors it covers, composed container, settled)`, assigned as
        #: one immutable tuple so a concurrent reader sees either the old state or the new, both
        #: valid. `settled` says no mint can move a resolution it made, which is what lets a later
        #: registration fold onto it instead of remaking.
        self._memo: tuple[int, int, Varied, bool] | None = (
            None if weight is None else (session._mint_epoch, 1, weight, True)
        )
        # §2.3e's ORIGINATION handle for the ambient: the context that last CHANGED the factor
        # list, which is NOT the one that happens to read it first. Stamping the reader would make
        # a `graphed.weight(parent)` call observable in a later divergence check.
        self._origin: EventContext = self
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

    def _selection_bridge(self) -> Any:
        """§9.1's `graphed.selection(ctx)` — the FULL three-case bridge, NOT `_selection()`.

        Case 1 (mask-derived): the mask that derived this context, skipping `vary` identity links.
        Case 2 (universe/nominal-derived): that label's member of the PARENT's selection — an
        unvaried `Array` in the GRANDparent's row space, `None` when the parent is root (where
        `_selection()` returns `None`, the row-space/context reason §6.4a's universe/nominal REFUSE
        control relies on). Case 3 (root): `None`.
        """
        node: EventContext | None = self
        while node is not None and node._link is not None:
            kind, payload = node._link
            if kind == "mask":
                return payload
            if kind == "project":
                parent_selection = node._parent._selection() if node._parent is not None else None
                return None if parent_selection is None else member_of(parent_selection, payload)
            node = node._parent  # `vary` identity link: skip
        return None

    def _ambient_weight(self) -> Varied | Array | None:
        """The composed ambient weight, `None` while nothing is registered; a projection's is the
        one member it adopted (§2.2).

        A cache read: the memo answers when it still stands at the Session's mint epoch, and any
        mint since it was stamped makes this REMAKE the composition from the original factors, so
        every factor is resolved against the registry as of this read. The factor list is never
        replaced, which is what makes the remake possible and what makes inserting or removing a
        `graphed.weight()` read change no composed value.
        """
        if not self._factors:
            return None
        if len(self._factors) == 1 and not isinstance(self._factors[0], Varied):
            # §2.2: a projection adopted ONE resolved member and nothing registered after it, so
            # there is no universe left to resolve — composing would only wrap it in a one-label
            # container, and the read's type would then depend on what minted since the projection
            adopted: Array = self._factors[0]
            return adopted
        epoch = self._session._mint_epoch
        memo = self._memo
        if memo is not None and memo[0] == epoch and memo[1] == len(self._factors):
            return memo[2]
        composed, settled = self._materialise(self._ambient_operands())
        self._memo = (epoch, len(self._factors), composed, settled)
        return composed

    def _foldable(self) -> tuple[int, int, Varied, bool] | None:
        """The memo the next composition may build ON, `None` when it must remake from the
        original factors.

        One predicate for both readers — the operand list below and `_vary_weight`'s fold — so the
        record-time check walks exactly what the read will compose, in every memo state. A memo a
        mint has passed is stale; an UNSETTLED one is a composition a later mint can still move an
        operand of, and folding onto it would freeze a resolution the read remakes.
        """
        memo = self._memo
        if memo is None or memo[0] != self._session._mint_epoch or not memo[3]:
            return None
        return memo

    def _ambient_operands(self) -> list[Any]:
        """What the next composition multiplies: the composed container plus whatever registered
        after it while the memo may be built on, and the original factors otherwise.
        """
        memo = self._foldable()
        return list(self._factors) if memo is None else [memo[2], *self._factors[memo[1] :]]

    def _materialise(self, operands: Sequence[Any]) -> tuple[Varied, bool]:
        """Compose `operands` over the recorded union, with the verdict on whether a later mint
        could still move any resolution it just made (`_settled` over every operand and label)."""
        composed = rebuild(
            _compose(operands, self._recorded), tags=self._ambient_tags(), context=self._origin
        )
        stamped = accessors.with_context(stamp_labels(composed), self._origin)
        return stamped, _all_settled(self._session, operands, self._recorded)

    def _ambient_tags(self) -> dict[str, tuple[str, ...]]:
        """The ambient's §1.1 tag map, accumulated over the factors in registration order — what
        today's composed container carries, read without composing. A projection's factor is a
        bare member carrying no map, which is how a projection still drops it."""
        merged: dict[str, tuple[str, ...]] = {}
        for factor in self._factors:
            merged.update(getattr(factor, "_tags", None) or {})
        return merged

    def _context_labels(self) -> tuple[str, ...]:
        """§2.2's §2.4-ordered union: (a) the ambient registry's labels, (b) the labels of the
        `Varied` collections this context CARRIES, (c) the labels of its selection."""
        out: dict[str, None] = dict.fromkeys(self._recorded)
        for source in (*self._collections.values(), self._selection()):
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
        if self._factors:
            ambient = self._ambient_weight()
            child._adopt_ambient(child._stamp(expand(lambda value, on: value[on], (ambient, mask), {})))
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
        if self._factors:
            child._adopt_ambient(child._stamp(member_of(self._ambient_weight(), label)))
        child._record = child._stamp(child._record)
        self._projected[label] = child
        return child

    def _adopt_ambient(self, composed: Any) -> None:
        """A row-space change MATERIALISES: the child starts from the one composed container, which
        is also its running form state. Re-indexing every factor instead would cost factors x
        labels per derivation — at the sizes this composition exists for, the same order as the
        whole quadratic it removes.
        """
        self._factors = [composed]
        self._memo = (self._session._mint_epoch, 1, composed, True)
        self._recorded = _union(("nominal",), labels_of(composed))
        self._origin = self


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
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
    tags: Mapping[str, Any],
) -> EventContext:
    if ctx._is_data:
        raise GraphedError(
            f"variation {name!r} cannot be registered on a data context: data fills nominal-only, "
            "and accepting a registration whose labels the fill then drops would be a silent drop"
        )
    if is_weight:
        return _vary_weight(
            ctx, name, nominal, variations, collections, points, composes_as_union, max_universes, tags
        )
    return _vary_shift(
        ctx, name, nominal, variations, collections, points, composes_as_union, max_universes, tags
    )


def _carriers(ctx: EventContext) -> tuple[Any, ...]:
    """§4.11-4's carrier list for the context forms — the three `_context_labels` already reads.

    The ambient contributes the `(session, labels)` pair, not a container: both readers want labels
    and points only, so composing here would record nodes for a walk that never does arithmetic.
    """
    return (AmbientCarrier(ctx._session, ctx._recorded), *ctx._collections.values(), ctx._selection())


def _child_of(ctx: EventContext) -> EventContext:
    """A `vary` link: the row space is unchanged, only registrations differ (§6.1d kind (2)).

    So the ambient state is COPIED, never composed: seeding the child with the composed parent
    would make every registration a two-element fold and reproduce the quadratic exactly.
    """
    child = EventContext(
        ctx._session,
        ctx._record,
        is_data=ctx._is_data,
        parent=ctx,
        link=("vary", None),
        collections=ctx._collections,
    )
    child._factors = list(ctx._factors)
    child._recorded = ctx._recorded
    child._memo = ctx._memo
    child._origin = ctx._origin
    child._record = child._stamp(child._record)
    return child


def _vary_weight(
    ctx: EventContext,
    name: str,
    central: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
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
    ambient_tags = ctx._ambient_tags()
    inherited = ambient_tags.get(name, ())
    # What the next composition will multiply, decided HERE, before this registration mints, and by
    # the same predicate the read uses: onto a foldable memo the new factor folds (today's
    # two-element chain step, so a program reading at every intermediate context pays today's node
    # count and no more); otherwise the composition is remade from the original factors and this
    # registration joins them.
    base = ctx._ambient_operands()
    folds = ctx._foldable() is not None
    # a nuisance the ambient weight registers AS A WEIGHT (`old._tags`) is stacked: the composition
    # below resolves it label-aligned into the union via `_two_level(old, ...)`, so fanning it out
    # would double-count it. It is excluded from the discriminator (§2 stacked-weight case). A
    # nuisance the ambient merely CARRIES as labels — a shift leaked in, its `_tags` empty (§8-g) —
    # is a genuine dependency the member reads, and still fans out.
    composed = frozenset(ambient_tags)
    one_at_a_time, joints = gather_members(
        name,
        tags,
        variations,
        inherited,
        points,
        session=ctx._session,
        carriers=_carriers(ctx),
        composes_as_union=composes_as_union,
        max_universes=max_universes,
        composed=composed,
    )
    # §1.1's within-the-container clause, keyed by family NAME over the three carriers
    # `_context_labels` reads. `_members` alone is the §2.4 union and cannot say which family a
    # label came from; the same `name` is the correlated case (one knob, §2.1) and is admitted —
    # `check_family` refuses a repeated tag within it — while any OTHER family already spelling
    # this label would make one universe differ from nominal in two knobs.
    registered = {
        f"{n}_{t}"
        for tag_map in (
            ambient_tags,
            *((getattr(source, "_tags", None) or {}) for source in _carriers(ctx)[1:]),
        )
        for n, ts in tag_map.items()
        if n != name
        for t in ts
    }
    for label in one_at_a_time:
        if label in registered:
            raise GraphedError(f"variation label {label!r} is already carried by this container")
    # a machine-minted joint joins the factor container as a flat cross node so `_two_level` reads it
    # by name, but it is a cross-coordinate, not a tag of this family, so it stays out of `_tags`.
    factors = {"nominal": central, **one_at_a_time, **joints}
    check_members(factors)
    # §2.1(b)'s ROW-SPACE rule: an ancestor-handled factor is re-indexed across the intervening
    # links; a descendant or divergent one is a construction-time error naming the direction.
    factors = {label: accessors.reindex_to(factor, ctx) for label, factor in factors.items()}
    factor = rebuild(factors, tags={name: inherited + _tags_of(name, one_at_a_time)}, context=ctx)
    # the §2.4 union is RECORDED here, never recomputed at the read: recomputing it from the
    # context would hand the ambient every shift registered after it, and recomputing it from the
    # factors would drop the shift labels a factor computed on shifted objects is read through.
    recorded = _union(ctx._context_labels(), tuple(factors))
    # the record-time type check, run BEFORE anything is recorded so a refused registration leaves
    # no trace at all
    _check_forms(ctx._session, [*base, factor], recorded)
    record_labels(factor)  # §2.5's vary-time half; the members are stamped when they compose
    # §2.5's shift-after-weight operand one: this factor's OWN member node ids, by value.
    ctx._session._weight_factors.append((name, _member_nodes(factor)))

    child = _child_of(ctx)
    child._factors.append(factor)
    child._recorded = recorded
    child._origin = child
    if folds:
        folded, settled = child._materialise([*base, factor])
        child._memo = (ctx._session._mint_epoch, len(child._factors), folded, settled)
    else:
        child._memo = None
    return child


def _vary_shift(
    ctx: EventContext,
    name: str,
    nominal: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any]] | None,
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
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
            "points= is not accepted in the shift form; its tags are the INNER keys of the "
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
        one_at_a_time, joints = gather_members(
            name,
            inner,
            None,
            inherited,
            points,
            session=ctx._session,
            carriers=_carriers(ctx),
            composes_as_union=composes_as_union,
            max_universes=max_universes,
        )
        existing = dict(current._members) if isinstance(current, Varied) else {"nominal": current}
        # §4.6: a supplied member is projected by the label's own point, not flattened to its
        # central universe — which is what makes a shift (x) shift joint point expressible; a
        # machine-minted joint is already the cross node, so `member_of` on it is the identity
        resolved = {label: member_of(member, label) for label, member in {**one_at_a_time, **joints}.items()}
        check_members({**existing, **resolved})
        resolved = {label: accessors.reindex_to(member, ctx) for label, member in resolved.items()}
        for label in resolved:
            if label in existing:
                raise GraphedError(f"variation label {label!r} already varies {collection_name!r}")
        tag_map = dict(current._tags) if isinstance(current, Varied) else {}
        # a joint is a cross-coordinate, not a tag of this single family, so it stays out of `_tags`
        tag_map[name] = inherited + _tags_of(name, one_at_a_time)
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
    composed ambient weight is therefore always FLAT.

    Memoised on the Session, because the answer is asked again at every later registration: the
    record-time check re-runs the composition over the whole factor list, and §4.6's resolution
    (a point restriction and a scan of the container's registered points) is not a dict lookup.

    What is stored is the resolved LABEL PATH, never the member: a member is an array carrying its
    context, and a context holds its factors, so a stored member roots the whole analysis in the
    live Session and the page's finalizer never fires. On a hit the member is rebuilt from the live
    container by two dict lookups, which is what the memo was buying in the first place.
    """
    if not isinstance(container, Varied):
        return container
    session = session_of(container)
    if session is None:
        return member_of(member_of(container, label), label)
    page = session._universes.get(id(container))
    if page is None:
        page = session._universes[id(container)] = {}
        weakref.finalize(container, session._universes.pop, id(container), None)
    path = page.get(label)
    if path is not None:
        member = container._members[path[0]]
        return member if path[1] is None else member._members[path[1]]
    outer_key = container._key_for(label)
    member = container._members[outer_key]
    inner_key = member._key_for(label) if isinstance(member, Varied) else None
    if _settled(container, label):
        page[label] = (outer_key, inner_key)
    return member if inner_key is None else member._members[inner_key]


def _settled(container: Any, label: str) -> bool:
    """Whether a later mint could still move `_two_level(container, label)` — the one thing that
    makes the memo above safe to keep across registrations.

    §4.6 reads the point registry at each level, and every read that finds NOTHING is an answer
    the next `vary` can change: an unregistered label falls to the central universe, and an
    unregistered own label is off the container's axes, so a label minted later can start
    resolving onto it. Those are recomputed rather than stored. A label the container carries
    outright never reads the registry at all, and `"nominal"` is registered nowhere by construction.
    """
    return _fixed(container, label) and _fixed(member_of(container, label), label)


def _all_settled(session: Any, operands: Sequence[Any], labels: Sequence[str]) -> bool:
    """Whether every resolution the composition just made is one no later mint can move.

    Read off `_two_level`'s memo rather than recomputed: that memo stores an answer exactly when
    `_settled` holds, so a page missing a label is that label's unsettled resolution. It must be
    read straight after the composition — a mint can settle a label the composition read before it.
    A composition that is not settled must not be FOLDED onto, because a mint could move an operand
    it already multiplied; it is remade from the original factors instead.
    """
    return all(
        label in session._universes.get(id(operand), ())
        for operand in operands
        if isinstance(operand, Varied)
        for label in labels
    )


def _fixed(value: Any, label: str) -> bool:
    if not isinstance(value, Varied):
        return True
    if label in value._members:
        return True
    registry = point_registry(value)
    return label in registry and all(own in registry for own in value._members if own != "nominal")


# ---- §4's composition: linear in the registered factors ------------------------------------
def _member(member: Any, label: str) -> Any:
    """The composition's own operand at a label: the member itself."""
    return member


def _times(left: Any, right: Any, label: str) -> Any:
    """The composition's multiply, naming the LABEL it failed at.

    Almost every clash is refused at registration by the walk over forms below, which names the
    label the same way. The one that reaches here is a clash a later mint created at a label no
    registration could walk (§7), and without the label the message says only that some multiply in
    some universe is ill-typed.
    """
    try:
        return left * right
    except GraphedError as exc:
        raise GraphedTypeError(
            "mul", capture(), f"the ambient weight at variation label {label!r}: {exc}"
        ) from exc


def _compose(
    factors: Sequence[Any],
    labels: Sequence[str],
    project: Callable[[Any, str], Any] = _member,
    mul: Callable[[Any, Any, str], Any] = _times,
) -> dict[str, Any]:
    """The ambient at every label: the product of each factor's member for that label, read two
    levels deep so a factor computed on shifted objects contributes in the label's own universe.

    The association is chosen so the work shared between labels is built once — a balanced product
    tree over the nominal members, a complement pushdown handing each index the product of all the
    others, and a tree walk for a label two or more factors vary at. `D`, the set of varying
    indices, is decided by MEMBER IDENTITY: a factor's `_tags` never mentioning a label says
    nothing about whether its member there is the nominal one.

    `project` and `mul` make this one walk serve both times it is needed: over arrays when the
    ambient materialises, and over FORMS at registration (`_check_forms`). The backend's multiply
    inference is not associative for raise/no-raise, so a check that MODELLED this walk instead of
    running it would admit clashes the composition then refuses at the read.
    """
    centrals = [_two_level(factor, "nominal") for factor in factors]
    central_ids = [_member_nodes(central) for central in centrals]
    count = len(centrals)
    tree = _product_tree([project(central, "nominal") for central in centrals], mul)
    complements = _complements(tree, count, mul)
    composed: dict[str, Any] = {}
    for label in labels:
        applied = [_two_level(factor, label) for factor in factors]
        varying = [i for i in range(count) if _member_nodes(applied[i]) != central_ids[i]]
        if not varying:
            rest = tree[(0, count)]
        elif len(varying) == 1:
            rest = complements[varying[0]]
        else:
            rest = _outside(tree, count, varying, mul)
        parts = [project(applied[i], label) for i in varying]
        composed[label] = _product(parts if rest is None else [rest, *parts], mul, label)
    return composed


def _product(parts: Sequence[Any], mul: Callable[[Any, Any, str], Any], label: str) -> Any:
    product = parts[0]
    for part in parts[1:]:
        product = mul(product, part, label)
    return product


def _product_tree(centrals: Sequence[Any], mul: Callable[[Any, Any, str], Any]) -> dict[tuple[int, int], Any]:
    """`(lo, hi)` -> the product of `centrals[lo:hi]`, split at the midpoint. `N-1` multiplies,
    shared by every label, and a pure function of the registration order."""
    tree: dict[tuple[int, int], Any] = {}

    def build(lo: int, hi: int) -> Any:
        if hi - lo > 1:
            mid = (lo + hi) // 2
            tree[(lo, hi)] = mul(build(lo, mid), build(mid, hi), "nominal")
        else:
            tree[(lo, hi)] = centrals[lo]
        return tree[(lo, hi)]

    if centrals:
        build(0, len(centrals))
    return tree


def _complements(
    tree: Mapping[tuple[int, int], Any], count: int, mul: Callable[[Any, Any, str], Any]
) -> list[Any]:
    """`out[i]` = the product of every nominal but the `i`-th, in O(N) multiplies shared down the
    tree — so a label exactly one factor varies at costs one more. `None` at `count == 1`, where
    the product of everything else is empty."""
    out: list[Any] = [None] * count

    def push(lo: int, hi: int, outside: Any) -> None:
        if hi - lo == 1:
            out[lo] = outside
            return
        mid = (lo + hi) // 2
        left, right = tree[(lo, mid)], tree[(mid, hi)]
        push(lo, mid, right if outside is None else mul(outside, right, "nominal"))
        push(mid, hi, left if outside is None else mul(outside, left, "nominal"))

    if count:
        push(0, count, None)
    return out


def _outside(
    tree: Mapping[tuple[int, int], Any],
    count: int,
    varying: Sequence[int],
    mul: Callable[[Any, Any, str], Any],
) -> Any:
    """The maximal subtrees holding none of `varying`, folded left to right — `None` when the
    varying indices cover every leaf."""
    parts: list[Any] = []

    def walk(lo: int, hi: int) -> None:
        if not any(lo <= index < hi for index in varying):
            parts.append(tree[(lo, hi)])
        elif hi - lo > 1:
            mid = (lo + hi) // 2
            walk(lo, mid)
            walk(mid, hi)

    walk(0, count)
    return _product(parts, mul, "nominal") if parts else None


# ---- the same walk, over forms: the record-time refusal -------------------------------------
def _check_forms(session: Any, factors: Sequence[Any], labels: Sequence[str]) -> None:
    """Type-check the composition by RUNNING it over forms, node-free (§5).

    A cross-factor clash is the one thing the composition itself can raise, and `check_members`
    cannot see it — it compares a factor only against its own nominal. Running the walk here keeps
    that refusal inside `vary`'s §4.5 transactional scope, where the minted labels still roll back;
    deferred to the first read it would leave them bound for the Session's life.
    """
    _compose(
        factors,
        labels,
        lambda member, label: _member_form(session, member, label),
        lambda left, right, label: _mul_form(session, left, right, label),
    )


def _member_form(session: Any, member: Any, label: str) -> Any:
    """The form of a member the composition multiplies.

    `_two_level` resolves §2.2's one legal level of nesting, so an ordinary factor arrives here as
    an array. One that varies past that level would compose into an ambient universe that is
    itself a container — which no consumer can read, `graphed.universe(...).node_id` included — so
    it is refused here, inside `vary`, rather than at the read that trips over it.
    """
    if isinstance(member, Varied):
        raise GraphedTypeError(
            "mul",
            capture(),
            f"the ambient weight at variation label {label!r}: this factor's member is itself a "
            f"container over {list(member._members)}, one universe deeper than a weight factor nests",
        )
    return session.form(member)


def _mul_form(session: Any, left: Any, right: Any, label: str) -> Any:
    """The product form of two member forms, memoised on the Session (`_mul_forms`).

    The walk multiplies one factor's handful of member forms at a time, so the distinct pairs stay
    few whatever the union's size; without the memo it would pay a backend inference per multiply.
    The key is the `Form` protocol's own `describe()` and NEVER `str(form)`: a repr may abbreviate
    — awkward's elides the middle of a deep type — so two forms that multiply differently would
    share a key and the memo would admit the very clash the walk exists to refuse.
    """
    key = (left.describe(), right.describe())
    hit = session._mul_forms.get(key)
    if hit is not None:
        return hit
    try:
        form = session.backend.op_form("mul", [left, right], {})
    except Exception as exc:  # the clash the composition would raise, named at its own label
        raise GraphedTypeError(
            "mul", capture(), f"the ambient weight at variation label {label!r}: {exc}"
        ) from exc
    session._mul_forms[key] = form
    return form


def _union(*groups: Sequence[str]) -> tuple[str, ...]:
    out: dict[str, None] = {"nominal": None}
    for group in groups:
        for label in group:
            out.setdefault(label, None)
    return tuple(out)


def _tags_of(name: str, members: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(label[len(name) + 1 :] for label in members if label != "nominal")
