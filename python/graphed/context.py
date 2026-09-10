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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from . import accessors
from ._kinds import Kind
from ._points import restrict
from ._tags import canonical_tag
from .array import Array
from .by_label import cone
from .errors import GraphedError, GraphedTypeError
from .provenance import capture
from .varied import (
    Varied,
    expand,
    labels_of,
    member_of,
    point_registry,
    rebuild,
    registered_points,
    session_of,
)
from .vary import (
    AmbientCarrier,
    _member_nodes,
    check_members,
    gather_members,
    record_labels,
    register,
    stamp_labels,
)

#: contexts are compared by IDENTITY (§2.6b), so a divergence error must name two distinguishable
#: objects; the serial plus the user line where the context was built is that name
_SERIAL = itertools.count()
_SLOT = itertools.count()

#: ("mask", mask) derives rows, ("vary", None) only registrations, ("project", label) narrows to
#: one universe — §6.1d's three link kinds, in one place
Link = tuple[str, Any]


@dataclass(frozen=True, slots=True)
class Rider:
    """§2.3's provenance of ONE ambient operation, kept BESIDE its slot and NEVER read off the
    entry — which a re-index replaces with a bare member, carrying no tags and no role.

    ``graphed.context.ambient_entries`` hands these out: what the ambient is made of, where each
    operation came from, and every row space it has been carried through since.
    """

    #: ``"factor"`` (multiplies) or ``"overlay"`` (replaces at its family's labels): §2.1's two kinds
    kind: str
    #: the families this entry carries, as ``{name: coordinates}`` — the container's tag map when
    #: it was registered, EXTENDED at a join. What decides ownership of a projected label, which
    #: factors an overlay covers, and which families a widening names
    families: Mapping[str, tuple[str, ...]]
    #: the entry's nominal node in each row space it came through, oldest first (§2.3): a central
    #: built where the entry came from still names it after an expansion re-indexed it
    priors: tuple[Any, ...] = ()
    #: an OVERLAY's prefix — the factor slots it was read over, which it covers
    prefix: tuple[int, ...] = ()
    #: the context that registered it, or that last re-indexed it
    home: Any = None
    #: every row-space link it has been re-indexed through since, oldest first
    links: tuple[Link, ...] = ()
    #: an overlay re-indexed INTO its own universe: there every value is that universe, so the
    #: composition replaces with it at every label rather than at the ones its family spans
    fixed: bool = False


@dataclass(frozen=True, slots=True)
class _Registration:
    """§2.7: one `graphed.vary` call, recorded where it happened so `explain` can say how each
    source of uncertainty entered. What the ambient KEEPS is the entry list and its riders; a
    registration is a fact about the call, and no rule reads one."""

    name: str
    kind: Kind
    tags: tuple[str, ...]
    context: Any
    form: str  # "new" | "join" | "overlay" | "shift"
    entered: str


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
        "_adopted", "_collections", "_derived", "_factors", "_gens", "_head", "_is_data",
        "_link", "_memo", "_origin", "_overlays", "_parent", "_projected",
        "_provenance", "_reads", "_record", "_recorded", "_registration", "_riders", "_serial",
        "_session", "_slots", "_weight_tags",
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
        #: the SLOT of each entry, minted at append and kept by a join, so a handle recorded over
        #: `(pu, hf)` still names those operations after `lf` joined the `hf` container — which
        #: `id()` could not, the join replacing the object the read was recorded over.
        self._slots: list[int] = [] if weight is None else [next(_SLOT)]
        #: whether `_factors[0]` is the composed container a row-space change ADOPTED, and so
        #: stands for the parent's own factors (`_live_factors` walks back through it)
        self._adopted = False
        #: §2.3: what that adopted head STANDS FOR — the parent's live factor slots and their
        #: generations at the adoption, the same tuple a read records. A record from across the
        #: change names the head when it equals this, and names nothing the child can anchor
        #: otherwise (a prefix of the parent's list, or a read from after that list grew).
        self._head: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        #: §2.1's ORDERED operations: a SLOT named here holds an OVERLAY — a relative-delta
        #: family whose members are the whole ambient, so the composition REPLACES the running
        #: value at its labels instead of multiplying.
        self._overlays: frozenset[int] = frozenset()
        #: §2.1's record of what a `graphed.weight()` READ handed out: `(the container's member
        #: nodes, the live FACTOR slots it composed, their generations)`. A central naming one of
        #: these is that composition, and the overlay lands right after the factors it was read
        #: over. The key is the WHOLE member map, not the nominal: a join leaves the nominal node
        #: untouched, so two handles read either side of one are told apart by nothing else.
        #: Overlays are not in the key: they never move the nominal, so inserting one among the
        #: factors leaves every earlier handle naming the same composition. SHARED down a `vary`
        #: link (one row space, so every id means the same value), replaced by a row-space change.
        self._reads: list[_Read] = []
        #: §2.1's staleness stamp: `slot -> generation`, bumped by a join whose union ADDS
        #: universes to that slot's nominal member. A handle read before such a join is no longer
        #: the composition it names, and a re-index (which changes every node but no universe)
        #: must not look like one — which is why this counts unions and not nodes.
        self._gens: dict[int, int] = {}
        #: §2.3's RIDERS: `slot -> Rider`, the provenance of each operation this context's list
        #: introduced, beside the slot and never on the entry. The ancestors' stay reachable
        #: through the adoption, exactly as their entries do (`_live_riders`).
        self._riders: dict[int, Rider] = (
            {}
            if weight is None
            else {self._slots[0]: Rider("factor", dict(getattr(weight, "_tags", None) or {}), home=self)}
        )
        #: the weight families REGISTERED on this lineage, `{name: tags}` — the record `variations`
        #: and a same-name registration read, as opposed to the ambient container's tag map, which
        #: a row-space change widens with every shift the mask carries
        self._weight_tags: dict[str, tuple[str, ...]] = {} if parent is None else dict(parent._weight_tags)
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
        #: §2.7: the `graphed.vary` call that produced THIS context, `None` for every other one.
        #: `explain` walks the lineage for them, which is what puts the families in registration
        #: order and names the context each was registered on.
        self._registration: _Registration | None = None
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
            self._record_read(adopted)
            return adopted
        epoch = self._session._mint_epoch
        memo = self._memo
        if memo is not None and memo[0] == epoch and memo[1] == len(self._factors):
            self._record_read(memo[2])
            return memo[2]
        composed, settled = self._materialise(self._ambient_operands())
        self._memo = (epoch, len(self._factors), composed, settled)
        self._record_read(composed)
        return composed

    def _record_read(self, composed: Any) -> None:
        """§2.1: the READ is what lets a later central name this composition. What is recorded is
        what a later registration compares — the member nodes the handle carries, the live factor
        slots it stands for and their generations — never the container, which a mint may remake.
        """
        slots = _live_slots(self)
        entry = (_member_nodes(composed), slots, tuple(self._gens.get(slot, 0) for slot in slots))
        if entry not in self._reads:
            self._reads.append(entry)

    def _overlay_ids(self) -> frozenset[int]:
        """The overlay slots as the OBJECT ids `_compose` tests, so the composition walk keeps
        working on the fold base, where the head is a composed container in no slot."""
        return _marks(self._factors, self._slots, self._overlays)

    def _fixed_ids(self) -> frozenset[int]:
        """The same, for the overlays a projection into their own universe FIXED (§2.3): there
        every value is that universe, so the composition replaces with them at every label."""
        return _marks(
            self._factors,
            self._slots,
            frozenset(slot for slot, rider in self._riders.items() if rider.fixed),
        )

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
            _compose(operands, self._recorded, overlays=self._overlay_ids(), fixed=self._fixed_ids()),
            tags=self._ambient_tags(),
            context=self._origin,
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
        child._weight_tags = {}  # a projection drops the registry (§2.2)
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
        self._slots = [next(_SLOT)]
        self._adopted = True
        self._overlays = frozenset()
        # the child's OWN reads start empty; the ancestors' stay reachable through the adoption
        # (`_lineage_reads`), because the head is exactly the composition they recorded (§2.3)
        self._reads = []
        parent = self._parent
        slots = _live_slots(parent) if parent is not None else ()
        gens = parent._gens if parent is not None else {}
        self._head = (slots, tuple(gens.get(slot, 0) for slot in slots))
        # the live list is the adopting parent's, so its stamps come along
        self._gens = dict(gens)
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
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None,
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
    factors = _lineage_factors(ctx)

    def resolve(label: str) -> frozenset[int]:
        # A factor whose member at `label` is its nominal carries nothing of the label (the seed
        # weight, a family with no coordinate there): reading it is not composition at `label`.
        out: set[int] = set()
        for factor in factors:
            out.update(
                set(_member_nodes(_two_level(factor, label)))
                - set(_member_nodes(_two_level(factor, "nominal")))
            )
        return frozenset(out)

    ambient = AmbientCarrier(ctx._session, ctx._recorded, resolve)
    return (ambient, *ctx._collections.values(), ctx._selection())


def _lineage_factors(ctx: EventContext) -> tuple[Any, ...]:
    """Every registered factor along the context's ancestry, once each. A row-space change adopts
    the ONE composed container as the child's factor, so a member computed from an ancestor's
    ambient reads the ancestor's factors, never the adopted node (§2.1(b))."""
    seen: dict[int, Any] = {}
    node: EventContext | None = ctx
    while node is not None:
        for factor in node._factors:
            seen.setdefault(id(factor), factor)
        node = node._parent
    return tuple(seen.values())


def _live_factors(ctx: EventContext) -> tuple[list[Any], list[int], frozenset[int]]:
    """The entries `ctx`'s ambient weight is composed FROM, in registration order, with their
    slots and the slots of the overlays among them.

    A row-space change adopts the ONE composed container, so the entries it stands for are the
    adopting parent's own live list; walking back through the adoption is what lets a weight
    registered on a mask-derived child name a factor of its parent (§2.1). Not `_lineage_factors`,
    which answers with every factor ever registered on the ancestry: one that an extension has
    since replaced is no longer multiplied in, and naming it would double it.
    """
    chain: list[EventContext] = []
    node: EventContext | None = ctx
    while node is not None:
        chain.append(node)
        if not node._adopted:
            break
        node = node._parent
    # by id, root-first: a `vary` child copies its parent's list, so the same entry is met again
    # at every context below the one that registered it
    seen: dict[int, tuple[Any, int]] = {}
    overlays: frozenset[int] = frozenset()
    for context in reversed(chain):
        overlays |= context._overlays
        start = 1 if context._adopted else 0
        for factor, slot in zip(context._factors[start:], context._slots[start:], strict=True):
            seen.setdefault(id(factor), (factor, slot))
    live = [factor for factor, _slot in seen.values()]
    slots = [slot for _factor, slot in seen.values()]
    return live, slots, overlays & frozenset(slots)


def _live_riders(ctx: EventContext) -> dict[int, Rider]:
    """The RIDER of every live slot, over the same chain `_live_factors` walks (§2.3).

    Deepest first, so the context that last re-indexed a slot answers with the whole chain it
    built. The links a row-space change has not yet carried the entry through — a child adopts its
    parent's composition and re-indexes nothing until something expands it — are added here, so a
    rider always reports where its entry stands as read from `ctx`.
    """
    riders: dict[int, Rider] = {}
    node: EventContext | None = ctx
    while node is not None:
        for slot, rider in node._riders.items():
            riders.setdefault(slot, rider)
        if not node._adopted:
            break
        node = node._parent
    return {slot: _pending(rider, ctx) for slot, rider in riders.items()}


def _pending(rider: Rider, ctx: EventContext) -> Rider:
    """`rider` with the row-space links between where its entry stands and `ctx` appended: a `vary`
    link changes no row space and is left out, as it is everywhere else."""
    home = rider.home
    links = () if home is None else tuple(link for link in ctx._links_below(home) if link[0] != "vary")
    return rider if not links else replace(rider, links=(*rider.links, *links))


def _spans(overlay: Any, label: str) -> bool:
    """Whether an OVERLAY replaces the running value at `label` — `_compose_ordered`'s own test, so
    a projection re-indexes an entry to exactly what the ambient at that label reads of it."""
    return _member_nodes(_two_level(overlay, label)) != _member_nodes(
        _two_level(overlay, "nominal")
    ) and not _placed_elsewhere(overlay, label)


def _fixes(ctx: EventContext, entry: Any, rider: Rider) -> str | None:
    """The universe between the entry's row space and `ctx` that FIXES this overlay — its own,
    where every value is that universe — or `None` (a factor, or an overlay `ctx` is not inside).
    """
    if rider.kind != "overlay":
        return None
    if rider.fixed:
        return next((payload for kind, payload in reversed(rider.links) if kind == "project"), "")
    home = rider.home
    if home is None:
        return None
    return next(
        (
            payload
            for kind, payload in ctx._links_below(home)
            if kind == "project" and payload != "nominal" and _spans(entry, payload)
        ),
        None,
    )


def _reindex_entry(ctx: EventContext, entry: Any, rider: Rider) -> tuple[Any, bool]:
    """§2.3's re-index of one ENTRY (never of a user value): what the ambient at each universe on
    the way down READS of it, decided from the rider.

    Through a mask, each label's member by that label's mask, as everything else. Through a
    projection to `L`, the TWO-LEVEL member at `L` — a factor computed on shifted objects carries
    its dependence on a shift label one level down, so the one-level read that is right for the
    composed head would silently drop it. An OVERLAY is re-indexed into its own universe and marked
    FIXED (there every value is that universe); one with no coordinate on `L` contributes nothing
    there and is left out, which `None` says.
    """
    home = accessors.context_of(entry)
    if home is None or home is ctx:
        return entry, rider.fixed
    value, fixed = entry, rider.fixed
    for kind, payload in ctx._links_below(home):
        if kind == "project" and payload != "nominal":
            if rider.kind == "overlay":
                if not _spans(value, payload):
                    return None, False
                fixed = True
            value = _two_level(value, payload)
        else:
            value = accessors._follow(value, kind, payload)
    return accessors.with_context(value, ctx), fixed


def _expanded(ctx: EventContext) -> tuple[list[Any], list[int], frozenset[int], dict[int, Rider]]:
    """The ancestor's live operations re-indexed to `ctx`, order, kinds and slots preserved — the
    expansion a join of an ancestor's factor, or an overlay anchored inside the head, performs.

    Each entry goes through `_reindex_entry`, so an overlay a projection leaves nothing of is left
    out and the rest keep the nominal they had in every row space they came through.
    """
    live, slots, overlays = _live_factors(ctx)
    riders = _live_riders(ctx)
    entries: list[Any] = []
    kept: list[int] = []
    carried: dict[int, Rider] = {}
    for entry, slot in zip(live, slots, strict=True):
        rider = riders[slot]
        value, fixed = _reindex_entry(ctx, entry, rider)
        if value is None:
            continue
        entries.append(value)
        kept.append(slot)
        carried[slot] = replace(
            rider, priors=(*rider.priors, _two_level(entry, "nominal")), home=ctx, fixed=fixed
        )
    return entries, kept, overlays & frozenset(kept), carried


def _live_slots(ctx: EventContext) -> tuple[int, ...]:
    """The slots of the live PRODUCT factors, in registration order — what a read records and what
    an overlay's prefix is measured against. Overlays are left out: one inserted among the factors
    changes no nominal, so a handle read before it still names the same composition (§2.1)."""
    _live, slots, overlays = _live_factors(ctx)
    return tuple(slot for slot in slots if slot not in overlays)


def _prefix_slots(
    ctx: EventContext, slots: Sequence[int], at: int, overlays: frozenset[int], adopted: bool
) -> tuple[int, ...]:
    """The FACTOR slots an overlay inserted at position `at` covers — the ones it was read over.

    An adopted head stands for the ancestor's whole live list, so an overlay anchored after it
    covers exactly the slots the head stands for, plus whatever this context registered before the
    anchor; once the head is expanded the slots are the list's own.
    """
    head = ctx._head[0] if adopted and ctx._head is not None else ()
    own = tuple(slot for slot in slots[1 if adopted else 0 : at] if slot not in overlays)
    return (*head, *own)


def ambient_entries(ctx: EventContext) -> tuple[tuple[int, Rider, Any], ...]:
    """The live ambient weight as `(slot, rider, entry)` records, oldest first (§2.3).

    The instrument for what the ambient is MADE of: each operation, its kind, the families it
    carries, the nominal it had in every row space it came through, and the links it was carried
    through to get here. Read from the RIDERS beside the slots, never off the entries, which a
    re-index leaves as bare members carrying neither tags nor role.

    Deliberately not re-exported from `graphed`: it reads the mechanism, and the analysis idiom is
    `graphed.weight` / `graphed.variations`.
    """
    if not isinstance(ctx, EventContext):
        raise GraphedError(
            "graphed.context.ambient_entries reads an event context's ambient weight operations"
        )
    live, slots, _overlays = _live_factors(ctx)
    riders = _live_riders(ctx)
    return tuple((slot, riders[slot], entry) for entry, slot in zip(live, slots, strict=True))


@dataclass(frozen=True, slots=True)
class Family:
    """§2.7: one ``graphed.vary`` registration, as ``graphed.explain`` reports it."""

    #: the family name and its coordinates
    name: str
    kind: Kind
    tags: tuple[str, ...]
    #: the registering context's links from the root — ``("cut", "universe:hf_up", "vary", …)``
    at: tuple[str, ...]
    #: how it entered: the collections a shift form varies, or the weight form's outcome
    entered: str
    #: ``(label, point)`` for each universe this family PLACED at a point of other coordinates
    placements: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    #: the WEIGHT families whose coordinates appear in none of this one's minted labels — the ones
    #: it COMPOSES with rather than fanning out over, which is why their joint is not a universe
    composes_with: tuple[str, ...]
    #: the SHIFT families a minted label of this one carries a coordinate of: the shifted objects
    #: this family read, so its members fan out over them
    fans_out_over: tuple[str, ...]
    #: the SHIFT families no minted label of this one names — it does not read those objects. A
    #: shift is never COMPOSED with: reading it or not reading it are the only two cases.
    independent_of: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Operation:
    """§2.7: one live ambient operation, read from its rider."""

    slot: int
    kind: str
    families: tuple[tuple[str, tuple[str, ...]], ...]
    #: the row-space links this entry was carried through, oldest first
    links: tuple[str, ...]
    #: the entry's member node ids. The RECORD carries them; the text never prints them, which is
    #: what makes the rendering byte-identical across two Sessions of one program.
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Variation:
    """§2.7: one universe this context carries, and where it came from."""

    label: str
    origin: str
    #: the families whose coordinates the label's registered point names
    families: tuple[str, ...]
    point: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Explanation:
    """§2.7: how a user's sources of uncertainty became this context's variations.

    ``str()`` renders one line per item under three headings: the families in registration order,
    the ambient's operations oldest first, and the universes carried here with their origins.
    """

    families: tuple[Family, ...]
    operations: tuple[Operation, ...]
    variations: tuple[Variation, ...]

    def __str__(self) -> str:
        lines = [
            f"graphed.explain: {len(self.families)} registrations, "
            f"{len(self.operations)} ambient operations, {len(self.variations)} universes",
            "families (registration order)",
        ]
        for family in self.families:
            where = "/".join(family.at) or "the root"
            placed = "".join(
                f", placing {label} at {_render_point(point)}" for label, point in family.placements
            )
            relations = "".join(
                f"; {phrase} {', '.join(names)}"
                for phrase, names in (
                    ("fans out over", family.fans_out_over),
                    ("independent of", family.independent_of),
                    ("composes with", family.composes_with),
                )
                if names
            )
            lines.append(
                f"  {family.name} ({family.kind.name}) {list(family.tags)} at {where}: "
                f"{family.entered}{placed}{relations}"
            )
        lines.append("ambient operations (oldest first)")
        # POSITION, not the slot: slots come from a process-global counter, so a second Session's
        # would differ while the composition is the same one. The record keeps the slot.
        for position, operation in enumerate(self.operations):
            carries = ", ".join(f"{name}{list(tags)}" for name, tags in operation.families)
            # An entry the registering context still holds has crossed no row space, which is a
            # fact about it, not a missing field — so it is said rather than left as a placeholder.
            through = f"via {', '.join(operation.links)}" if operation.links else "registered here"
            lines.append(f"  #{position} {operation.kind}: {carries} {through}")
        lines.append("universes here")
        for variation in self.variations:
            lines.append(f"  {variation.label}: {variation.origin}")
        return "\n".join(lines)


def _render_point(point: Sequence[tuple[str, str]]) -> str:
    return "{" + ", ".join(f"{name}: {tag}" for name, tag in point) + "}"


def _link_name(link: Link) -> str:
    kind, payload = link
    if kind == "mask":
        return "cut"
    if kind == "project":
        return "nominal" if payload == "nominal" else f"universe:{payload}"
    return "vary"


def _path_from_root(ctx: EventContext) -> tuple[str, ...]:
    """The links from the root down to `ctx`, as names (§2.7).

    A RUN of `vary` links is collapsed to one: they change no row space, so a run of them says
    only how many registrations came before, which the family list already orders. Every
    row-space link is kept as it stands, including two cuts in a row.
    """
    root: EventContext = ctx
    while root._parent is not None:
        root = root._parent
    path: list[str] = []
    for link in ctx._links_below(root):
        name = _link_name(link)
        if name != "vary" or not path or path[-1] != "vary":
            path.append(name)
    return tuple(path)


def _lineage_registrations(ctx: EventContext) -> tuple[_Registration, ...]:
    """Every `graphed.vary` call on this lineage, oldest first (§2.7)."""
    found: list[_Registration] = []
    node: EventContext | None = ctx
    while node is not None:
        if node._registration is not None:
            found.append(node._registration)
        node = node._parent
    return tuple(reversed(found))


def explain(ctx: EventContext) -> Explanation:
    """§2.7: how this context's variations came to be — the families, the ambient's operations and
    the universes carried here, as a frozen record whose `str()` is one line per item.

    Everything is DERIVED, never re-decided: the families from the registrations the lineage
    recorded, the operations from the riders (`ambient_entries`), and each universe's origin from
    its registered point against those families. The ambient is read exactly as `graphed.weight`
    reads it and nothing else is composed, so a call mints no node a `weight` read would not.
    """
    if not isinstance(ctx, EventContext):
        raise GraphedError("graphed.explain reads an event context")
    registrations = _lineage_registrations(ctx)
    ambient = ctx._ambient_weight()
    carried: dict[str, None] = dict.fromkeys(labels_of(ambient))
    for collection in ctx._collections.values():
        carried.update(dict.fromkeys(labels_of(collection)))
    points = ctx._session._points
    variations: list[Variation] = []
    minted: dict[str, set[str]] = {registration.name: set() for registration in registrations}
    for label in carried:
        if label == "nominal":
            continue
        point = tuple((name, tag) for name, tag in points.get(label, ()))
        variations.append(_origin_of(label, point, registrations))
    for variation in variations:
        for name in variation.families:
            minted.setdefault(name, set()).add(variation.label)
    kinds: dict[str, Kind] = {}
    for registration in registrations:
        kinds.setdefault(registration.name, registration.kind)
    relations = {
        registration.name: _relations(registration, kinds, minted, variations)
        for registration in registrations
    }
    families = tuple(
        Family(
            registration.name,
            registration.kind,
            registration.tags,
            _path_from_root(registration.context),
            registration.entered,
            tuple(
                (variation.label, variation.point)
                for variation in variations
                if variation.origin.startswith(f"{registration.name} placed")
            ),
            *relations[registration.name],
        )
        for registration in registrations
    )
    operations = tuple(
        Operation(
            slot,
            rider.kind,
            tuple((name, tags) for name, tags in rider.families.items()),
            tuple(_link_name(link) for link in rider.links),
            _member_nodes(entry),
        )
        for slot, rider, entry in ambient_entries(ctx)
    )
    return Explanation(families, operations, tuple(variations))


def _origin_of(
    label: str, point: tuple[tuple[str, str], ...], registrations: Sequence[_Registration]
) -> Variation:
    """§2.7: where one universe came from, read off its registered point and the family whose
    coordinate spells it — never by re-deciding anything."""
    families = tuple(dict.fromkeys(name for name, _tag in point))
    own = next(
        (r for r in registrations for tag in r.tags if label == f"{r.name}_{tag}"),
        None,
    )
    if own is not None:
        others = tuple(name for name in families if name != own.name)
        if others:  # the placing family's own axis is dropped from a placement's point (§2.1)
            return Variation(
                label, f"{own.name} placed at {_render_point(point)}", (own.name, *others), point
            )
        if own.form == "overlay":
            return Variation(label, f"{own.name}'s own universe, a relative-delta family", (own.name,), point)
        return Variation(label, f"{own.name}, one at a time", (own.name,), point)
    joint = next(
        (r for r in registrations for tag in r.tags if label.startswith(f"{r.name}_{tag}__")),
        None,
    )
    if joint is not None:
        read = tuple(name for name in families if name != joint.name)
        return Variation(label, f"{joint.name} fanned out over {', '.join(read) or '?'}", families, point)
    return Variation(label, f"a point over {', '.join(families) or 'no registered family'}", families, point)


def _relations(
    registration: _Registration,
    kinds: Mapping[str, Kind],
    minted: Mapping[str, set[str]],
    variations: Sequence[Variation],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """§2.7: what one WEIGHT family does with every other family, read off the labels it minted.

    Two weight families whose coordinates never share a label COMPOSE: their joint is a product,
    not a universe, which is why `hf_up__mu_up` is not in the list. A shift is never composed
    with — a weight family either read the shifted objects, so a minted label carries that
    shift's coordinate and its members FAN OUT over it, or it did not and is INDEPENDENT of it.
    A shift family's own line reports none of the three: the relation is a weight family's.
    """
    if registration.kind != Kind.WEIGHT:
        return (), (), ()
    reached = {
        other
        for variation in variations
        if variation.label in minted.get(registration.name, ())
        for other in variation.families
    }
    return (
        tuple(
            name
            for name, kind in kinds.items()
            if kind == Kind.WEIGHT and name != registration.name and name not in reached
        ),
        tuple(name for name, kind in kinds.items() if kind == Kind.SHIFT and name in reached),
        tuple(name for name, kind in kinds.items() if kind == Kind.SHIFT and name not in reached),
    )


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
    child._slots = list(ctx._slots)
    child._adopted = ctx._adopted
    child._head = ctx._head
    child._overlays = ctx._overlays
    # the same list object, not a copy: one row space, so a read at either end names the same
    # values, and a handle read from the parent after the child was built still decides here
    child._reads = ctx._reads
    child._gens = dict(ctx._gens)
    child._riders = dict(ctx._riders)
    child._recorded = ctx._recorded
    child._memo = ctx._memo
    child._origin = ctx._origin
    child._record = child._stamp(child._record)
    return child


_Read = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def _lineage_reads(ctx: EventContext) -> Iterator[_Read]:
    """Every read record `ctx` can name — its own and every ancestor row space's.

    Unlike `_live_factors`, this does NOT stop where the adoption ended: a family that joined an
    ancestor's factor expands the head into that ancestor's operations re-indexed here, keeping
    their slots and generations, so a handle read before the change still names a prefix of the
    live factor slots and is matched like any other record (§2.3). The records are keyed by the
    ancestor's member nodes, which is what a crossed handle carries.

    A `vary` link shares the list object, so a list already yielded is skipped.
    """
    node: EventContext | None = ctx
    seen: set[int] = set()
    while node is not None:
        if id(node._reads) not in seen:
            seen.add(id(node._reads))
            yield from node._reads
        node = node._parent


def _extension(ctx: EventContext, central: Any) -> tuple[str, Any] | None:
    """§2.1: what a weight registration's central NAMES — a composition a read handed out
    (`("ambient", the index the overlay goes at)`, or `("prefix", the slot it was read over)` when
    that index is inside an adopted head the registration must expand first), a composition a later
    join has since WIDENED (`("stale", the families that grew)`, which `_vary_weight` refuses), an
    already-registered factor (`("factor", its slot)`), or nothing new.

    Three answers are refusals `_vary_weight` raises on (§2.3): `("owned", the universe)` for a
    factor the universe this context is projected into is OF, and `("covered", …)` /
    `("covered-read", …)` for an operation that would land among the factors a relative-delta
    family covers, inside that family's own universe — a join of one of them, or a handle read over
    part of them.

    By NODE, never by value: a re-computed expression with equal values is a new factor. The READ
    is asked FIRST: a handle over ONE factor carries that factor's own nominal, so the factor arm
    would claim it and union the handle's labels into the factor's nominal member — the same nodes
    under new coordinates, a widening in name only that moves the slot's generation and fires both
    §2.1 refusals on programs §2.1 admits.
    """
    node = member_of(central, "nominal")  # ONE level in, not `_two_level`, which peels a second
    if isinstance(node, Varied):  # nested past §2.2's one level; `_check_forms` names it properly
        return None
    live, slots, overlays = _live_factors(ctx)
    riders = _live_riders(ctx)
    # a composition a READ recorded, WITHOUT composing anything here: its FACTORS must still be a
    # prefix of the live factors, which is what makes the handle a rescaling of the operations it
    # was built from and lets the overlay land right after them. Reads are recorded per row space,
    # so the member nodes alone answer which composition the handle is.
    factors = tuple(slot for slot in slots if slot not in overlays)
    key = _member_nodes(central)
    stale: list[int] = []
    for members, read, stamps in _lineage_reads(ctx):
        if members != key or read != factors[: len(read)]:
            continue
        moved = [slot for slot, was in zip(read, stamps, strict=True) if ctx._gens.get(slot, 0) != was]
        if moved:  # a join has since ADDED universes under this handle (§2.1)
            stale = moved
            continue
        at = _overlay_index(ctx, read)
        if at is not None and read[-1] not in ctx._slots:
            # the record is the adopted head's own tuple: the overlay anchors right after it and
            # nothing is re-indexed, so no operation lands inside a covered prefix (§2.3)
            return ("ambient", at)
        if not read:
            continue
        covered = _fixed_cover(ctx, read[-1], live, slots, riders)
        if covered is not None:
            return ("covered-read", covered)
        if at is not None:
            return ("ambient", at)
        # the prefix ends inside the adopted head, which the registration expands (§2.3)
        return ("prefix", read[-1])
    if stale:
        widened = {family for slot in stale if slot in riders for family in riders[slot].families}
        return ("stale", ", ".join(sorted(widened)))
    # an OVERLAY's nominal is the ambient's own nominal, so it would answer this test in the
    # factor's place; only a product factor names a factor
    for factor, slot in zip(live, slots, strict=True):
        if slot in overlays:
            continue
        rider = riders[slot]
        # the entry as it stands here, then what it was in each row space it came through: an
        # expansion re-indexed it, and the central names the node of the space it was built in
        for candidate in (_two_level(factor, "nominal"), *rider.priors):
            if not _same_node(candidate, node):
                continue
            owned = _owned_projection(ctx, candidate, rider)
            if owned is not None:
                return ("owned", owned)
            covered = _fixed_cover(ctx, slot, live, slots, riders)
            return ("factor", slot) if covered is None else ("covered", covered)
    return None


def _owned_projection(ctx: EventContext, candidate: Any, rider: Rider) -> str | None:
    """§2.3: the universe projected into between the candidate's row space and `ctx` that this
    entry OWNS — a family of its RIDER is a coordinate of that universe's point — or `None`.

    A projection to another universe keeps the entries' identity (it re-indexes the composed
    product, not the nodes a central names), which is what lets a central built above it name its
    factor there. For the factor the universe is OF, that is a trap: its member at the label IS the
    universe projected into, so joining would put the central back and erase it. A joint label is
    owned by every family in its point, which is why the point decides and not the label's name —
    and the families come from the rider, because the re-index the expansion performs leaves a bare
    member behind, with no tag map for this to read.
    """
    home = accessors.context_of(candidate)
    if home is None or not home._is_ancestor_of(ctx):
        return None
    registry = ctx._session._points
    return next(
        (
            payload
            for kind, payload in ctx._links_below(home)
            if kind == "project"
            and payload != "nominal"
            and any(name in rider.families for name, _tag in registry.get(payload, ()))
        ),
        None,
    )


def _fixed_cover(
    ctx: EventContext,
    slot: int,
    live: Sequence[Any],
    slots: Sequence[int],
    riders: Mapping[int, Rider],
) -> tuple[str, str] | None:
    """§2.3: the `(family, universe)` of an overlay that covers `slot` and is FIXED at `ctx` —
    projected into its own universe, where its member is the whole ambient rescaled and every
    value is that universe.

    An operation landing inside the factors such an overlay covers has nothing to re-derive it
    over: the overlay's member is the user's node over the OLD product. `None` when no overlay
    covers the slot, or when the one that does still composes as an overlay.
    """
    for entry, entry_slot in zip(live, slots, strict=True):
        rider = riders.get(entry_slot)
        if rider is None or slot not in rider.prefix:
            continue
        label = _fixes(ctx, entry, rider)
        if label is not None:
            return next(iter(rider.families), rider.kind), label
    return None


def _overlay_index(ctx: EventContext, read: tuple[int, ...]) -> int | None:
    """Where in `ctx`'s OWN entry list an overlay over `read` belongs — right after the last FACTOR
    it was composed over and after any overlay already anchored there, so two relative-delta
    families over the same factors stay in registration order. `None` when that factor is not this
    list's to order (a prefix ending inside an ancestor's entries, which no read here records)."""
    if not read:
        return None
    if read[-1] in ctx._slots:
        at = ctx._slots.index(read[-1]) + 1
    elif ctx._adopted and ctx._head is not None and read == ctx._head[0]:
        # the adopted head stands for exactly this prefix, so the overlay goes after it and the
        # ancestor's operations stay folded inside it — nothing is re-indexed
        at = 1
    else:
        return None
    while at < len(ctx._slots) and ctx._slots[at] in ctx._overlays:
        at += 1
    return at


def _same_node(left: Any, right: Any) -> bool:
    """Whether two values are the same IR node, read anywhere along ONE lineage.

    The ids answer for almost every pair, so they are compared first; a MASK between the handles
    then makes one id two different values, and so does a projection to any universe but `nominal`
    — there each entry re-indexes to its member at that label, which is not the node the central
    matched. A `vary` link and `graphed.nominal(ctx)` keep both the row space and the identity, so
    a central re-derived at the nominal projection names the parent's factor.
    """
    if left.node_id != right.node_id:
        return False
    here, there = accessors.context_of(left), accessors.context_of(right)
    if here is None or there is None or here is there:
        return True
    deep, shallow = (here, there) if there._is_ancestor_of(here) else (there, here)
    if not shallow._is_ancestor_of(deep):
        return False
    return not any(
        kind == "mask" or (kind == "project" and payload != "nominal")
        for kind, payload in deep._links_below(shallow)
    )


def _extend(
    ctx: EventContext,
    slot: int,
    members: Mapping[str, Any],
    name: str,
    family: tuple[str, ...],
) -> tuple[Varied, list[Any], list[int], frozenset[int], bool, dict[int, Rider]]:
    """§2.1's joins-a-factor outcome as `(the container the family joined, the new entry list, its
    slots, the overlay slots in it, whether the union WIDENED the joined nominal member, the
    riders)`, written so the named container is multiplied in exactly ONCE.

    A factor of an ANCESTOR replaces the composed container this context adopted by the entries it
    stands for, re-indexed here — the one place that pays §2.1(b)'s factors x labels, and only for
    the context that asked. The joined container keeps its SLOT, so order and kind are preserved: a
    family joining a factor registered before an overlay keeps that factor's position and the
    overlay still replaces the product of the operations it was read over.
    """
    covering = _covering_overlay(ctx, slot)
    if slot in ctx._slots:
        at = ctx._slots.index(slot)
        joined, widened = _joined(ctx._factors[at], members, name, family, ctx, covering)
        entries = list(ctx._factors)
        entries[at] = joined
        # a join keeps the nominal node, so the entry stands for what it always did
        riders = {**ctx._riders, slot: _carries(ctx._riders[slot], name, family)}
        return joined, entries, list(ctx._slots), ctx._overlays, widened, riders
    entries, slots, overlays, riders = _expanded(ctx)
    at = slots.index(slot)
    joined, widened = _joined(entries[at], members, name, family, ctx, covering)
    entries[at] = joined
    riders[slot] = _carries(riders[slot], name, family)
    return joined, entries, slots, overlays, widened, riders


def _carries(rider: Rider, name: str, family: tuple[str, ...]) -> Rider:
    """§2.3: a join EXTENDS the rider's families — the joined container carries the new one too,
    and every rule that reads them (ownership, an overlay's cover, a widening's message) must see
    it. The nominal node is untouched by a join, so the identity chain is unchanged."""
    return replace(rider, families={**rider.families, name: family})


def _covering_overlay(ctx: EventContext, slot: int) -> str | None:
    """The family of the first OVERLAY whose prefix covers `slot`, `None` when none does."""
    _live, slots, _overlays = _live_factors(ctx)
    riders = _live_riders(ctx)
    return next(
        (
            family
            for entry_slot in slots
            if (rider := riders.get(entry_slot)) is not None
            and rider.kind == "overlay"
            and slot in rider.prefix
            for family in rider.families
        ),
        None,
    )


def _marks(entries: Sequence[Any], slots: Sequence[int], overlays: frozenset[int]) -> frozenset[int]:
    """The overlay slots as the OBJECT ids `_compose` tests (see `EventContext._overlay_ids`)."""
    return frozenset(id(entry) for entry, slot in zip(entries, slots, strict=True) if slot in overlays)


def _overlay(ctx: EventContext, members: Mapping[str, Any], name: str, family: tuple[str, ...]) -> Varied:
    """§2.1's relative-delta family as an ORDERED operation on the ambient.

    Its NOMINAL is the ambient's own nominal node, which is what makes it an overlay rather than a
    factor: the composition REPLACES the running value with its member at the family's labels
    instead of multiplying, because that member already IS the whole ambient rescaled. Every other
    label reads its nominal, so it contributes nothing there and the factors compose as before.
    """
    return rebuild(
        {**members, "nominal": _two_level(members["nominal"], "nominal")},
        tags={name: family},
        context=ctx,
    )


def _joined(
    container: Any,
    members: Mapping[str, Any],
    name: str,
    family: tuple[str, ...],
    ctx: EventContext,
    covering: str | None = None,
) -> tuple[Varied, bool]:
    """`container` with the joining family's universes among its own members, over the UNION of the
    two centrals' coordinates (§2.1).

    The shared node stays the nominal, and every coordinate either central declares — a central
    computed over shifted objects IS the weight's dependence on that shift — is carried, so the
    ambient reads the weight at each label's own objects whichever family registered first. The
    same coordinate declared by both with different nodes is two weights, not one, and is refused
    here, before anything of this registration is recorded. A projection's adopted member is a bare
    array and becomes the nominal.
    """
    existing = container._members if isinstance(container, Varied) else {"nominal": container}
    added = {label: member for label, member in members.items() if label != "nominal"}
    tags = {**(getattr(container, "_tags", None) or {}), name: family}
    nominal, widened = _union_nominal(existing["nominal"], members["nominal"], name, ctx, covering)
    return rebuild({**existing, "nominal": nominal, **added}, tags=tags, context=ctx), widened


def _union_nominal(
    carried: Any, central: Any, name: str, ctx: EventContext, covering: str | None
) -> tuple[Any, bool]:
    """The joined container's NOMINAL: the shared node, carrying every coordinate either central
    declares. A central computed over shifted objects is a container whose universes ARE the
    weight's dependence on that shift, and `_two_level` reads them at the ambient's shift labels,
    so the union is what makes the ambient read the weight at each label's own objects whichever
    family registered first."""
    left = carried._members if isinstance(carried, Varied) else {"nominal": carried}
    right = central._members if isinstance(central, Varied) else {"nominal": central}
    for label, member in right.items():
        if label != "nominal" and label in left and _member_nodes(left[label]) != _member_nodes(member):
            raise GraphedError(
                f"graphed.vary({name!r}): its central names the weight factor already registered "
                f"here, but the two disagree at coordinate {label!r} — the registered central "
                f"carries node {_member_nodes(left[label])} there and this one carries "
                f"{_member_nodes(member)}; one weight cannot have two values at one coordinate"
            )
    merged = {**left, **right}
    widened = set(merged) > set(left)
    if widened and covering is not None:
        # §2.1: the overlay's members ARE the composition as it stood when its handle was read;
        # widening a factor under it would change that composition beneath them.
        raise GraphedError(
            f"graphed.vary({name!r}): its central names the weight factor registered here, but "
            f"joining would add the universes {sorted(set(merged) - set(left))} to it, and the "
            f"relative-delta family {covering!r} was registered on a graphed.weight() handle read "
            "over that factor; register the absolute family first and the relative-delta family "
            "on a handle read after it"
        )
    if len(merged) == 1:
        return merged["nominal"], widened
    tags = {**(getattr(carried, "_tags", None) or {}), **(getattr(central, "_tags", None) or {})}
    return rebuild(merged, tags=tags, context=ctx), widened


def _vary_weight(
    ctx: EventContext,
    name: str,
    central: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None,
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
    inherited = ctx._weight_tags.get(name, ())
    # What the next composition will multiply, decided HERE, before this registration mints, and by
    # the same predicate the read uses: onto a foldable memo the new factor folds (today's
    # two-element chain step, so a program reading at every intermediate context pays today's node
    # count and no more); otherwise the composition is remade from the original factors and this
    # registration joins them.
    base = ctx._ambient_operands()
    folds = ctx._foldable() is not None
    # §2.1: the central NAMES the factor it varies. Decided here, before anything is minted, and
    # read only for its answer — nothing is composed or re-indexed to reach it, so a program that
    # names nothing records exactly the nodes it records without this.
    match = _extension(ctx, central)
    if match is not None and match[0] == "stale":
        # §2.1: the handle names a composition a later join WIDENED, so its universes are not that
        # composition's any more. Refused here, before this registration mints anything at all.
        raise GraphedError(
            f"graphed.vary({name!r}): its central is a graphed.weight() handle read BEFORE a later "
            f"registration added universes to the weight factor it composed ({match[1]}), so the "
            "handle's universes are no longer that composition; read the handle again after that "
            "registration (`w = graphed.weight(ctx)`) and register this family on the new handle"
        )
    if match is not None and match[0] == "owned":
        # §2.3: this context IS that universe, and the named factor's member there is what makes it
        # one — joining would put the central back and the projection would vanish
        raise GraphedError(
            f"graphed.vary({name!r}): its central names the weight factor that the universe "
            f"{match[1]!r} this context is projected into is OF, whose member there is that "
            "universe rather than the central; register this family on the factor at the parent, "
            "before the projection, or re-derive the central from this context's collections, "
            "which names nothing and starts a new factor"
        )
    if match is not None and match[0] == "covered":
        # §2.3: inside the overlay's own universe its member is the whole ambient rescaled — the
        # user's node over the OLD product — and nothing is minted to re-derive it over a new one
        covered_family, universe_label = match[1]
        raise GraphedError(
            f"graphed.vary({name!r}): its central names a weight factor that the relative-delta "
            f"family {covered_family!r} was registered over, and this context is projected into "
            f"{universe_label!r}, that family's own universe, where its member IS the whole ambient "
            "and joining underneath it would change the composition its members already are; "
            "register this family on the factor at the parent, before the projection, or read a "
            "graphed.weight() handle AT this context and register the family on that"
        )
    if match is not None and match[0] == "covered-read":
        covered_family, universe_label = match[1]
        raise GraphedError(
            f"graphed.vary({name!r}): its central is a graphed.weight() handle read over a strict "
            f"prefix of the weight factors the relative-delta family {covered_family!r} was "
            f"registered over, and this context is projected into {universe_label!r}, that "
            "family's own universe, where nothing re-derives that composition; read the handle "
            "again AT this context (`w = graphed.weight(ctx)`) and register this family on it"
        )
    # The ambient's tag-map families are only CANDIDATES for composition (m56): a member's coordinate
    # on one of them is dropped iff the member's node at that label reads a lineage factor's varied
    # member there (`_reads_ambient` in `vary._foreign`), which the composition below would multiply
    # in again via `_two_level`; reached through shifted objects instead, it fans out. The map is
    # over-inclusive — a mask-derived child's adopted container carries leaked shifts — and that is
    # harmless because the node test decides.
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
    family = inherited + _tags_of(name, one_at_a_time)
    if match is not None and match[0] == "factor":
        # joining an existing factor REMAKES the composition from the new entry list: the fold memo
        # is built on the premise that a registration only ever appends
        factor, updated, slots, overlays, widened, riders = _extend(ctx, match[1], factors, name, family)
        joined_slot: int | None = match[1] if widened else None
        operands = updated
    else:
        joined_slot = None
        slot = next(_SLOT)
        entries, entry_slots, riders = ctx._factors, ctx._slots, ctx._riders
        adopted = ctx._adopted
        prefix: tuple[int, ...] = ()
        if match is None:
            factor = rebuild(factors, tags={name: family}, context=ctx)
            at, overlays, kind = len(ctx._factors), ctx._overlays, "factor"
        else:
            # §2.1: the OVERLAY lands right after the operations the handle it was built from was
            # read over, so a factor registered after that read multiplies its result
            factor = _overlay(ctx, factors, name, family)
            kind = "overlay"
            if match[0] == "prefix":
                # §2.3: the read ends INSIDE the adopted head, so the head is expanded into the
                # operations it stands for, re-indexed here exactly as a join of an ancestor's
                # factor expands it, and the overlay anchors among them
                entries, entry_slots, overlays, riders = _expanded(ctx)
                adopted = False
                at = entry_slots.index(match[1]) + 1
                while at < len(entry_slots) and entry_slots[at] in overlays:
                    at += 1
            else:
                at, overlays = match[1], ctx._overlays
            prefix = _prefix_slots(ctx, entry_slots, at, overlays, adopted)
            overlays = overlays | {slot}
        riders = {**riders, slot: Rider(kind, {name: family}, prefix=prefix, home=ctx)}
        updated = [*entries[:at], factor, *entries[at:]]
        slots = [*entry_slots[:at], slot, *entry_slots[at:]]
        # an entry at the END is an append the memo folds onto — the overlay replaces the folded
        # composition at its own labels, which is what its members already are
        operands = [*base, factor] if at == len(entries) else updated
    # the §2.4 union is RECORDED here, never recomputed at the read: recomputing it from the
    # context would hand the ambient every shift registered after it, and recomputing it from the
    # factors would drop the shift labels a factor computed on shifted objects is read through.
    recorded = _union(ctx._context_labels(), tuple(factors))
    if operands is not updated and not _resolvable(base, tuple(factors)):
        operands = updated  # the memo cannot answer a label this call minted (below)
    # the record-time type check, run BEFORE anything is recorded so a refused registration leaves
    # no trace at all
    _check_forms(
        ctx._session,
        operands,
        recorded,
        _marks(updated, slots, overlays),
        _marks(updated, slots, frozenset(s for s, rider in riders.items() if rider.fixed)),
    )
    if match is None:
        form, entered = "new", "a new factor"
    elif match[0] == "factor":
        others = sorted(set(riders[match[1]].families) - {name})
        form = "join"
        entered = f"joins the factor carrying {', '.join(others)}" if others else "joins a factor"
    else:
        covered = sorted(
            {
                carried
                for covered_slot in prefix
                for carried in (riders[covered_slot].families if covered_slot in riders else ())
            }
        )
        form = "overlay"
        entered = f"an overlay over {', '.join(covered) or 'the composition it was read over'}"
    record_labels(factor)  # §2.5's vary-time half; the members are stamped when they compose
    # §2.5's shift-after-weight operand one: this factor's OWN member node ids, by value.
    ctx._session._weight_factors.append((name, _member_nodes(factor)))

    child = _child_of(ctx)
    child._registration = _Registration(name, Kind.WEIGHT, family, ctx, form, entered)
    child._weight_tags[name] = family
    child._factors = updated
    child._slots = slots
    child._overlays = overlays
    child._riders = riders
    if joined_slot is not None:
        # §2.1's staleness stamp: every handle read over this factor before now composed universes
        # this join has just added to, and naming one of them is refused from here on
        child._gens[joined_slot] = ctx._gens.get(joined_slot, 0) + 1
    # the adoption marker outlives a registration only while the container the row-space change
    # adopted is still the head of the list an extension may expand back into its own factors
    child._adopted = ctx._adopted and updated[0] is ctx._factors[0]
    child._recorded = recorded
    child._origin = child
    # a join, and an overlay inserted before the end, remake the composition from the entry list;
    # only an append composed onto the memo above and can fold onto it (§3)
    if folds and operands is not updated:
        folded, settled = child._materialise(operands)
        child._memo = (ctx._session._mint_epoch, len(child._factors), folded, settled)
    else:
        child._memo = None
    return child


def _vary_shift(
    ctx: EventContext,
    name: str,
    nominal: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None,
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
            "collection mappings (or a Varied member's own tags), so pass "
            "collections={Name: {tag: record}} or collections={Name: varied}"
        )
    mapping: dict[str, Any] = dict(tags)
    for collection_name, inner in (collections or {}).items():
        if collection_name in mapping:
            raise GraphedError(f"collection {collection_name!r} was named twice")
        mapping[collection_name] = inner
    if not mapping:
        raise GraphedError(f"the shift form of graphed.vary({name!r}) needs at least one collection")
    for collection_name, inner in list(mapping.items()):
        if isinstance(inner, Varied):  # m55: lockstep by propagation, unpacked to the hand form
            mapping[collection_name] = _unpack_varied(ctx, name, collection_name, inner, points)
    _check_lockstep(name, mapping)

    child = _child_of(ctx)
    replaced = dict(ctx._collections)
    family: tuple[str, ...] = ()
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
        family = tag_map[name] = inherited + _tags_of(name, one_at_a_time)
        replaced[collection_name] = child._stamp(
            register(rebuild({**existing, **resolved}, tags=tag_map, context=ctx))
        )
        _report_shift_after_weight(ctx, collection_name, existing)
    child._collections = replaced
    child._registration = _Registration(
        name, Kind.SHIFT, family, ctx, "shift", f"shifts {', '.join(mapping)}"
    )
    return child


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


def _unpack_varied(
    ctx: EventContext,
    name: str,
    collection_name: str,
    varied: Varied,
    points: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """m55: a `Varied` collection member is accepted only when it carries EXACTLY the family being
    registered and its nominal is this context's own collection; it unpacks to the `{tag: member}`
    map the hand form passes, so nothing downstream changes. Everything else is refused here,
    before any label is minted."""
    hand = f"collections={{{collection_name!r}: {{tag: record}}}}"
    if points:
        raise GraphedError(
            f"points= placements are not accepted beside a Varied collection member ({collection_name!r}): "
            f"its members carry no fan-out to prune; pass {hand} for a placed registration"
        )
    tags = dict(varied._tags)
    expected = {"nominal", *(f"{name}_{tag}" for tag in tags.get(name, ()))}
    extra = sorted(set(labels_of(varied)) - expected)
    if set(tags) != {name} or extra:
        raise GraphedError(
            f"collection {collection_name!r}: a Varied member must carry exactly the family {name!r} being "
            f"registered, got families {sorted(tags)} with extra labels {extra}; build it as "
            f"graphed.vary(graphed.nominal(ctx[{collection_name!r}]), {name!r}, ...) on the context's central "
            f"collection with only the tags being added, or pass {hand}"
        )
    current = ctx._read(collection_name)
    central = member_of(current, "nominal") if isinstance(current, Varied) else current
    nominal = accessors.reindex_to(member_of(varied, "nominal"), ctx)
    if nominal.node_id != central.node_id:
        raise GraphedError(
            f"collection {collection_name!r}: the Varied member's nominal is node {nominal.node_id}, not this "
            f"context's {collection_name!r} (node {central.node_id}), so its members are not shifts of this "
            f"collection; build it on the context's central collection or pass {hand}"
        )
    return {tag: member_of(varied, f"{name}_{tag}") for tag in tags[name]}


def _check_lockstep(name: str, mapping: Mapping[str, Mapping[Any, Any]]) -> None:
    """§2.6a: all collections in one call MUST share one tag set (the lockstep Jet+MET form)."""
    sets = {}
    for collection_name, inner in mapping.items():
        if not isinstance(inner, Mapping):
            raise GraphedError(
                f"collection {collection_name!r} needs a {{tag: record}} mapping or a Varied, "
                f"got {type(inner).__name__}"
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
    overlays: frozenset[int] = frozenset(),
    fixed: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """The ambient at every label: the product of each factor's member for that label, read two
    levels deep so a factor computed on shifted objects contributes in the label's own universe.

    An OVERLAY among the entries makes the ambient an ordered sequence of operations rather than a
    product, so it takes the walk below instead of this one.

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
    if overlays:
        return _compose_ordered(factors, labels, project, mul, overlays, fixed)
    return _product_over(factors, labels, project, mul)


def _product_over(
    factors: Sequence[Any],
    labels: Sequence[str],
    project: Callable[[Any, str], Any],
    mul: Callable[[Any, Any, str], Any],
) -> dict[str, Any]:
    """The balanced product of `factors` at every label — `_compose` without the overlays, so a run
    of product operations between two overlays composes exactly as a whole list of them does. An
    empty run answers with no labels at all, which is how a leading or adjacent overlay reads."""
    if not factors:
        return {}
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


def _compose_ordered(
    factors: Sequence[Any],
    labels: Sequence[str],
    project: Callable[[Any, str], Any],
    mul: Callable[[Any, Any, str], Any],
    overlays: frozenset[int],
    fixed: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """§2.1's ORDERED operations, for a factor list carrying an overlay.

    Each RUN of product operations between two overlays composes with the same balanced tree the
    whole list takes when there is no overlay, so the ambient stays one tree walk per label plus
    one multiply per overlay — an overlay changes the shape of the composition, never its cost
    class. An OVERLAY then replaces the running value at the labels its family spans, because its
    members ARE the whole ambient rescaled: the operations up to it compose exactly to the node
    they were built from, so multiplying would count it twice. Whether an overlay spans a label is
    the same MEMBER-IDENTITY test the product walk uses, EXCEPT at a universe another family
    PLACED at a point carrying the overlay's coordinate (§2.1): the member declared there is the
    user's value for that point, and the join reads the same member two-level, so both outcomes
    agree there. An overlay a projection into its OWN universe FIXED replaces at every label
    instead, because there every value is that universe (§2.3). Order is registration order, so a
    factor registered after an overlay multiplies the overlay's result.
    """
    runs: list[dict[str, Any]] = []
    marks: list[Any] = []
    run: list[Any] = []
    for factor in factors:
        if id(factor) in overlays:
            runs.append(_product_over(run, labels, project, mul))
            marks.append(factor)
            run = []
        else:
            run.append(factor)
    runs.append(_product_over(run, labels, project, mul))
    nominal_ids = [_member_nodes(_two_level(overlay, "nominal")) for overlay in marks]
    composed: dict[str, Any] = {}
    for label in labels:
        running: Any = runs[0].get(label)
        for index, overlay in enumerate(marks):
            applied = _two_level(overlay, label)
            spans = id(overlay) in fixed or (
                _member_nodes(applied) != nominal_ids[index] and not _placed_elsewhere(overlay, label)
            )
            if running is None or spans:
                running = project(applied, label)
            part = runs[index + 1].get(label)
            if part is not None:
                running = part if running is None else mul(running, part, label)
        composed[label] = running
    return composed


def _resolvable(operands: Sequence[Any], labels: Sequence[str]) -> bool:
    """Whether the fold base can answer the labels this registration mints.

    §4.6 resolves a label by restricting its point to the container's OWN axes and matching an own
    label exactly. A COMPOSED container merges every factor's axes, so a point that reaches two of
    them (a placement, `{muR: 2, muF: 2}`) matches nothing and falls to the central universe, where
    the factor list restricts each axis to its own factor and finds it. Folding onto the base would
    freeze that miss — a placed universe reading its central value, and differently depending on
    whether a `weight()` read happened to leave a memo — so the composition is remade from the list.
    """
    for operand in operands:
        if not isinstance(operand, Varied):
            continue
        registry = point_registry(operand)
        carried = registered_points(operand)
        axes = frozenset(nuisance for own in carried.values() for nuisance, _ in own)
        for label in labels:
            point = registry.get(label)
            if point is None or label in operand._members:
                continue
            if restrict(point, axes) and operand._key_for(label) == "nominal":
                return False
    return True


def _placed_elsewhere(overlay: Any, label: str) -> bool:
    """§2.1: whether `label` is a universe ANOTHER family PLACED at a point carrying this overlay's
    coordinate — the tour's `scale_upup` over a `muR` factor and a relative-delta `muF`. The member
    the placing family declared there is the user's value for that point, and the join reads the
    same member two-level, so the overlay must leave it alone for the two outcomes to agree.

    `_route` drops the placing family's OWN axis from an additive placement's point, so a placed
    universe is exactly one that no coordinate of its own point names; a default one-at-a-time
    label and a fanned-out joint both carry theirs (`{name: tag}` and `{name: tag, **foreign}` under
    `f"{name}_{tag}"` and `f"{name}_{tag}__{foreign label}"`). The overlay's own universes are its
    members, placed or not, and it still replaces there.
    """
    if label in labels_of(overlay):
        return False
    point = point_registry(overlay).get(label)
    if point is None:
        return False
    return not any(label == f"{n}_{t}" or label.startswith(f"{n}_{t}__") for n, t in point)


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
def _check_forms(
    session: Any,
    factors: Sequence[Any],
    labels: Sequence[str],
    overlays: frozenset[int],
    fixed: frozenset[int] = frozenset(),
) -> None:
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
        overlays,
        fixed,
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
