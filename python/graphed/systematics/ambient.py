"""The ambient event weight (§2.1-§2.3): the operation list, its provenance, its composition.

An event context's `AmbientRecord` holds this state — the ordered operations, the slot ids, the
riders, the reads, the memo. This module alone builds and mutates it, and holds every rule that
reads it: which of the three outcomes a weight registration decides (a read it names, a factor it
joins, a new factor), how an entry is re-indexed across a row-space change, and how the operations
compose into one container. Moved out of `graphed.context` (m57): the context tree is one concern,
what rides it is another.

Every decision here reads PROVENANCE — the rider beside a slot, the recorded reads — and never the
entry object as it currently stands, which a re-index replaces with a bare member.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from ..array import Array
from ..errors import GraphedError, GraphedTypeError
from ..provenance import capture
from .by_label import cone
from .handles import follow, reindex_to, with_context
from .kinds import Kind
from .lineage import (
    ContextNode,
    ambient_of,
    collections_of,
    context_session,
    is_ancestor,
    link_of,
    links_below,
    parent_of,
    selection,
)
from .points import restrict
from .registration import (
    AmbientCarrier,
    check_members,
    gather_members,
    record_labels,
    register,
    stamp_labels,
)
from .tags import canonical_tag
from .varied import (
    Varied,
    context_of,
    key_for,
    labels_of,
    member_nodes,
    member_of,
    members_of,
    point_registry,
    rebuild,
    registered_points,
    session_of,
    tags_of,
)

if TYPE_CHECKING:
    from ..context import EventContext

_SLOT = itertools.count()


@dataclass(frozen=True, slots=True)
class Rider:
    """§2.3's provenance of ONE ambient operation, kept BESIDE its slot.

    It is NEVER read off the entry — which a re-index replaces with a bare member, carrying no tags
    and no role.

    ``graphed.context.ambient_entries`` hands these out: what the ambient is made of, where each
    operation came from, and every row space it has been carried through since.
    """

    #: ``"factor"`` (multiplies) or ``"overlay"`` (replaces at its family's labels): §2.1's two
    #: kinds
    kind: str
    #: the families this entry carries, as ``{name: coordinates}`` — the container's tag map when
    #: it was registered, EXTENDED at a join. What decides ownership of a projected label, which
    #: factors an overlay covers, and which families a widening names
    families: Mapping[str, tuple[str, ...]]
    #: the entry's nominal node in each row space it came through, oldest first (§2.3): a central
    #: built where the entry came from still names it after an expansion re-indexed it. A MASK
    #: makes one node two values, so crossing one adds an identity; a projection keeps it.
    priors: tuple[Any, ...] = ()
    #: the member the entry BECAME at each projected label it was carried through, ``{label:
    #: member}`` (§2.3). Recorded at the projection, before anything reads or expands it: at `L`
    #: the entry's member is what the ambient there is made of, and for the factor `L` is OF that
    #: member IS the universe, so a central node-identical to it names the entry as it stands
    #: there.
    projected: Mapping[str, Any] = field(default_factory=dict)
    #: an OVERLAY's prefix — the factor slots it was read over, which it covers
    prefix: tuple[int, ...] = ()
    #: the context that registered it, or that last re-indexed it
    home: Any = None
    #: the row-space links this operation came through, oldest first, as `(kind, label)` pairs —
    #: `("mask", None)`, `("project", "hf_up")`: the links the lineage took from the root down to
    #: where it stands. A `vary` link changes no row space and is not one of them.
    links: tuple[tuple[str, str | None], ...] = ()
    #: an overlay re-indexed INTO its own universe: there every value is that universe, so the
    #: composition replaces with it at every label rather than at the ones its family spans
    fixed: bool = False
    #: the universe that FIXED it, which every later row space reports and the refusals name — a
    #: further projection adds links, and the last one taken is not the one that fixed anything
    fixed_at: str | None = None


@dataclass(frozen=True, slots=True)
class Registration:
    """§2.7: one `graphed.vary` call, recorded where it happened.

    So `explain` can say how each source of uncertainty entered. What the ambient KEEPS is the
    entry list and its riders; a registration is a fact about the call, and no rule reads one.
    """

    name: str
    kind: Kind
    tags: tuple[str, ...]
    context: Any
    #: how it entered: `"factor"` (a new one), `"join"`, `"overlay"` or `"shift"` (§2.7(a))
    form: str
    #: the families the form NAMES — the other families on the factor a join entered, the families
    #: an overlay was read over; empty for a new factor and for a shift, which names collections
    #: instead
    names: tuple[str, ...] = ()
    #: the collections a SHIFT form varies — what pairs it with the §2.5 shift-after-weight
    #: registry
    varies: tuple[str, ...] = ()


_Read = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


class Memo(NamedTuple):
    """A context's composed ambient weight, cached (§5)."""

    #: the mint epoch it was composed at: any later mint makes the next read remake it
    epoch: int
    #: how many of the record's factors it covers
    covers: int
    composed: Varied
    #: no mint can move a resolution it made, which is what lets a registration fold onto it
    settled: bool


@dataclass(slots=True)
class AmbientRecord:
    """One context's ambient weight (§2.1-§2.3): its operations, their provenance, the cache.

    §2.6's ambient weight is a DEFERRED composition (§4): composing at every registration would be
    quadratic in the families registered, because each one rewrites every label the ambient
    already carries. Only this module constructs or mutates a record.
    """

    #: the registered operations in composition order
    factors: list[Any]
    #: the SLOT of each entry, minted at append and kept by a join, so a handle recorded over
    #: `(pu, hf)` still names those operations after `lf` joined the `hf` container
    slots: list[int]
    #: whether `factors[0]` is the composed container a row-space change ADOPTED, standing for the
    #: parent's own operations (`live_factors` walks back through it)
    adopted: bool
    #: §2.3: what that adopted head STANDS FOR — the parent's live factor slots and their
    #: generations at the adoption, the tuple a read records
    head: tuple[tuple[int, ...], tuple[int, ...]] | None
    #: §2.3: the operations the head stands for, re-indexed here with every join of one of them
    #: applied — `None` while nothing has joined one, where the ancestor's own live list answers
    head_ops: tuple[list[Any], list[int], frozenset[int]] | None
    #: §2.1's ORDERED operations: a slot named here holds an OVERLAY, which REPLACES the running
    #: value at its labels instead of multiplying
    overlays: frozenset[int]
    #: §2.1's record of what a `graphed.weight()` READ handed out: `(the container's member nodes,
    #: the live FACTOR slots it composed, their generations)`. Keyed by the whole member map, since
    #: a join leaves the nominal node untouched. SHARED down a `vary` link (one row space),
    #: replaced by a row-space change.
    reads: list[_Read]
    #: §2.1's staleness stamp, `slot -> generation`, bumped by a join whose union ADDS universes
    #: to that slot's nominal member; it counts unions, not nodes, so a re-index is not one
    gens: dict[int, int]
    #: §2.3's RIDERS, `slot -> Rider`, for the operations this context's list introduced
    riders: dict[int, Rider]
    #: the §2.4 label union RECORDED at each registration
    recorded: tuple[str, ...]
    #: the composition cache, assigned as one immutable value so a concurrent reader sees either
    #: the old state or the new
    memo: Memo | None
    #: §2.3e's ORIGINATION handle: the context that last CHANGED the operation list, not the one
    #: that happens to read it first; `None` is the context holding this record
    origin: ContextNode | None
    #: the weight families REGISTERED on this lineage, name to tags — what `variations` and a
    #: same-name registration read, unlike the ambient's tag map, which a mask widens
    weight_tags: dict[str, tuple[str, ...]]
    #: §2.7: the `graphed.vary` call that produced this context, `None` for every other one
    registration: Registration | None


def seed(ctx: ContextNode, weight: Varied | None, parent: ContextNode | None) -> AmbientRecord:
    """Return the record a context starts with: `weight` as its one factor, if any."""
    slots = [] if weight is None else [next(_SLOT)]
    riders = (
        {}
        if weight is None
        else {
            # a seeded context is a root: it came through no row-space link
            slots[0]: Rider(
                "factor", dict(tags_of(weight)), priors=(two_level(weight, "nominal"),), home=ctx
            )
        }
    )
    return AmbientRecord(
        factors=[] if weight is None else [weight],
        slots=slots,
        adopted=False,
        head=None,
        head_ops=None,
        overlays=frozenset(),
        reads=[],
        gens={},
        riders=riders,
        recorded=_union(("nominal",), labels_of(weight)),
        memo=(
            None
            if weight is None
            else Memo(context_session(ctx).systematics.epoch, 1, weight, True)
        ),
        origin=None,
        weight_tags={} if parent is None else dict(ambient_of(parent).weight_tags),
        registration=None,
    )


def for_vary_link(ctx: EventContext) -> AmbientRecord:
    """Return the record of a `vary` child of `ctx`: the same operations, COPIED, never composed.

    The row space is unchanged (§6.1d kind (2)); seeding the child with the composed parent would
    make every registration a two-element fold and reproduce the quadratic exactly.
    """
    held = ambient_of(ctx)
    return replace(
        held,
        factors=list(held.factors),
        slots=list(held.slots),
        gens=dict(held.gens),
        riders=dict(held.riders),
        origin=ctx if held.origin is None else held.origin,
        weight_tags=dict(held.weight_tags),
        registration=None,
    )


def adopt(
    child: EventContext, parent: EventContext, composed: Any, projected: str | None = None
) -> None:
    """Make `child`, across a row-space change, MATERIALISE `parent`'s ambient `composed`.

    Re-indexing every factor instead would cost factors x labels per derivation. A projection to
    `projected` also records each live entry's member there (§2.3) and drops the registry (§2.2).
    `composed` is `None` when `parent` has no ambient.
    """
    held = ambient_of(child)
    if composed is not None:
        slots = _live_slots(parent)
        gens = ambient_of(parent).gens
        held.factors = [composed]
        held.slots = [next(_SLOT)]
        held.adopted = True
        held.head_ops = None
        held.overlays = frozenset()
        # the child's OWN reads start empty; the ancestors' stay reachable through the adoption
        # (`_lineage_reads`), because the head is exactly the composition they recorded (§2.3)
        held.reads = []
        held.head = (slots, tuple(gens.get(slot, 0) for slot in slots))
        held.gens = dict(gens)
        held.memo = Memo(context_session(child).systematics.epoch, 1, composed, True)
        held.recorded = _union(("nominal",), labels_of(composed))
        held.origin = None
        # §2.1/§2.3: the adoption HANDS OUT the composed member, as a read does, so
        # `graphed.weight` and `graphed.explain` stay observations, not inputs
        _record_read(child, composed)
        if projected is not None and projected != "nominal":
            held.riders = project_riders(parent, projected)
    if projected is not None:
        held.weight_tags = {}


def weight_of(ctx: EventContext) -> Varied | Array | None:
    """Return the composed ambient weight, `None` while nothing is registered.

    A projection's is the one member it adopted (§2.2). The memo answers while it stands at the
    Session's mint epoch; a mint since makes this REMAKE the composition from the original
    factors, so every factor is resolved against the registry as of this read.
    """
    held = ambient_of(ctx)
    if not held.factors:
        return None
    if len(held.factors) == 1 and not isinstance(held.factors[0], Varied):
        # §2.2: a projection adopted ONE resolved member and nothing registered after it; the
        # adoption already recorded it as the read, so handing it out records nothing new
        adopted: Array = held.factors[0]
        return adopted
    epoch = context_session(ctx).systematics.epoch
    memo = held.memo
    if memo is not None and memo.epoch == epoch and memo.covers == len(held.factors):
        _record_read(ctx, memo.composed)
        return memo.composed
    composed, settled = _materialise(ctx, _ambient_operands(ctx))
    held.memo = Memo(epoch, len(held.factors), composed, settled)
    _record_read(ctx, composed)
    return composed


def fold(child: EventContext, operands: Sequence[Any]) -> None:
    """Cache `child`'s ambient as `operands` composed: an append onto the parent's memo (§3)."""
    held = ambient_of(child)
    folded, settled = _materialise(child, operands)
    held.memo = Memo(context_session(child).systematics.epoch, len(held.factors), folded, settled)


def context_labels(ctx: EventContext) -> tuple[str, ...]:
    """§2.2's §2.4-ordered union of the labels a context carries.

    (a) the ambient registry's, (b) those of the `Varied` collections it CARRIES, (c) those of its
    selection.
    """
    out: dict[str, None] = dict.fromkeys(ambient_of(ctx).recorded)
    for source in (*collections_of(ctx).values(), selection(ctx)):
        for label in labels_of(source):
            out.setdefault(label, None)
    return tuple(out)


def _origin(ctx: ContextNode) -> ContextNode:
    held = ambient_of(ctx).origin
    return ctx if held is None else held


def _record_read(ctx: EventContext, composed: Any) -> None:
    """§2.1: the READ is what lets a later central name this composition.

    What is recorded is what a later registration compares — the member nodes the handle carries,
    the live factor slots it stands for and their generations — never the container, which a mint
    may remake.
    """
    held = ambient_of(ctx)
    slots = _live_slots(ctx)
    entry = (member_nodes(composed), slots, tuple(held.gens.get(slot, 0) for slot in slots))
    if entry not in held.reads:
        held.reads.append(entry)


def _foldable(ctx: EventContext) -> Memo | None:
    """Return the memo the next composition may build ON, `None` when it must remake.

    One predicate for both the operand list below and the weight form's fold, so the record-time
    check walks exactly what the read will compose. A memo a mint has passed is stale; an
    UNSETTLED one could still move under a later mint, and folding onto it would freeze it.
    """
    memo = ambient_of(ctx).memo
    if memo is None or memo.epoch != context_session(ctx).systematics.epoch or not memo.settled:
        return None
    return memo


def _ambient_operands(ctx: EventContext) -> list[Any]:
    """Return what the next composition multiplies.

    The composed container plus whatever registered after it while the memo may be built on, and
    the original factors otherwise.
    """
    factors = ambient_of(ctx).factors
    memo = _foldable(ctx)
    return list(factors) if memo is None else [memo.composed, *factors[memo.covers :]]


def _materialise(ctx: EventContext, operands: Sequence[Any]) -> tuple[Varied, bool]:
    """Compose `operands` over the recorded union, and say whether a later mint could move it."""
    held = ambient_of(ctx)
    origin = _origin(ctx)
    fixed = frozenset(slot for slot, rider in held.riders.items() if rider.fixed)
    composed = rebuild(
        _compose(
            operands,
            held.recorded,
            overlays=_marks(held.factors, held.slots, held.overlays),
            fixed=_marks(held.factors, held.slots, fixed),
        ),
        tags=_ambient_tags(ctx),
        context=origin,
    )
    stamped = with_context(stamp_labels(composed), origin)
    return stamped, _all_settled(context_session(ctx), operands, held.recorded)


def _ambient_tags(ctx: EventContext) -> dict[str, tuple[str, ...]]:
    """Return the ambient's §1.1 tag map, accumulated over the factors in registration order.

    A projection's factor is a bare member carrying no map, which is how a projection drops it.
    """
    merged: dict[str, tuple[str, ...]] = {}
    for factor in ambient_of(ctx).factors:
        merged.update(tags_of(factor))
    return merged


def _carriers(ctx: EventContext) -> tuple[Any, ...]:
    """§4.11-4's carrier list for the context forms — the three `context_labels` already reads.

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
                set(member_nodes(two_level(factor, label)))
                - set(member_nodes(two_level(factor, "nominal")))
            )
        return frozenset(out)

    ambient = AmbientCarrier(context_session(ctx), ambient_of(ctx).recorded, resolve)
    return (ambient, *collections_of(ctx).values(), selection(ctx))


def _lineage_factors(ctx: EventContext) -> tuple[Any, ...]:
    """Every registered factor along the context's ancestry, once each.

    A row-space change adopts the ONE composed container as the child's factor, so a member
    computed from an ancestor's ambient reads the ancestor's factors, never the adopted node
    (§2.1(b)).
    """
    seen: dict[int, Any] = {}
    node: EventContext | None = ctx
    while node is not None:
        for factor in ambient_of(node).factors:
            seen.setdefault(id(factor), factor)
        node = parent_of(node)
    return tuple(seen.values())


def live_factors(ctx: EventContext) -> tuple[list[Any], list[int], frozenset[int]]:
    """Return the entries `ctx`'s ambient weight is composed FROM.

    In the order `_compose` applies them, with their slots and the slots of the overlays among
    them.

    The order is the LIST's: the record's `slots` hold every operation of this row space in the
    position the composition walks, and a row-space change adopts the ONE composed container, so
    the head stands for the adopting parent's own live list and expands in place. Walking back
    through the adoption is what lets a weight registered on a mask-derived child name a factor of
    its parent (§2.1). Not `_lineage_factors`, which answers with every factor ever registered on
    the ancestry: one that an extension has since replaced is no longer multiplied in, and naming
    it would double it.

    Never the order a lineage WALK meets the entries: an overlay is inserted after the factors it
    was read over, so a child registering a factor and then an overlay anchored before it is met in
    the opposite order from the one it composes in, and an expansion built that way would move the
    overlay past the factor and replace its product (§2.3).
    """
    held = ambient_of(ctx)
    entries = list(held.factors)
    slots = list(held.slots)
    overlays = held.overlays
    stood = _stood_for(ctx) if entries else None
    if stood is not None:
        above, above_slots, above_overlays = stood
        entries = [*above, *entries[1:]]
        slots = [*above_slots, *slots[1:]]
        overlays = overlays | above_overlays
    return entries, slots, overlays & frozenset(slots)


def _stood_for(ctx: EventContext) -> tuple[list[Any], list[int], frozenset[int]] | None:
    """Return the operations `ctx`'s adopted head stands for.

    `None` when its list is its own (§2.3).

    Once a join has extended one of them the head carries the join's universes and the operations
    it stands for are kept beside it (`_head_ops`), re-indexed here with that join applied — so a
    second family naming the same central, and every rider the projection records, read the JOINED
    operation rather than the ancestor's own.
    """
    held = ambient_of(ctx)
    if not held.adopted:
        return None
    if held.head_ops is not None:
        entries, slots, overlays = held.head_ops
        return list(entries), list(slots), overlays
    ancestor = _adopted_from(ctx)
    return None if ancestor is None else live_factors(ancestor)


def _adopted_from(ctx: EventContext) -> EventContext | None:
    """Return the context whose live list the adopted head at `ctx` stands for.

    The one the row-space link was taken from. A `vary` link shares the list, so the walk goes
    through it.
    """
    node: EventContext | None = ctx
    while node is not None and ((link := link_of(node)) is None or link[0] == "vary"):
        node = parent_of(node)
    return None if node is None else parent_of(node)


def _raw_riders(ctx: EventContext) -> dict[int, Rider]:
    """Return the rider of every live slot AS RECORDED.

    Over the same chain `live_factors` walks (§2.3).

    Deepest first, so the context that last recorded something about a slot — a re-index, or the
    member its entry became at a projected label — answers with the whole chain it built.
    """
    riders: dict[int, Rider] = {}
    node: EventContext | None = ctx
    while node is not None:
        for slot, rider in ambient_of(node).riders.items():
            riders.setdefault(slot, rider)
        if not ambient_of(node).adopted:
            break
        node = parent_of(node)
    return riders


def live_riders(ctx: EventContext) -> dict[int, Rider]:
    """Return the same riders with the links the entry has not yet been carried through.

    They are appended, so a rider always reports where its entry stands as read from `ctx`: a child
    adopts its parent's composition and re-indexes nothing until something expands it.
    """
    return {slot: _pending(rider, ctx) for slot, rider in _raw_riders(ctx).items()}


def project_riders(ctx: EventContext, label: str) -> dict[int, Rider]:
    """§2.3's provenance for a projection.

    Every live entry's MEMBER at `label`, recorded on the child the moment that universe comes to
    exist — before anything reads or expands there, so no later decision about it can depend on
    either.

    Nothing is minted, and nothing is re-indexed: a member is a lookup on a container already
    built, and the member is recorded as it stands in the entry's OWN row space — a mask between
    the entry and `ctx` is not walked, because walking one mints (§2.3), and the record is a NODE
    the central is compared against rather than a value anything composes.
    """
    live, slots, _overlays = live_factors(ctx)
    riders = _raw_riders(ctx)
    recorded: dict[int, Rider] = {}
    for entry, slot in zip(live, slots, strict=True):
        rider = riders[slot]
        member = two_level(_through_projections(ctx, entry), label)
        recorded[slot] = replace(rider, projected={**rider.projected, label: member})
    return recorded


def _through_projections(ctx: EventContext, entry: Any) -> Any:
    """`entry` as the ambient at `ctx` reads it through the PROJECTIONS between the two.

    Each one a member lookup, which mints nothing. A mask is left unwalked: its re-index would
    mint, and a decision mints nothing (§2.3), so the value stays in the row space above it.
    """
    home = context_of(entry)
    value = entry
    for kind, payload in () if home is None else links_below(ctx, home):
        if kind == "project" and payload != "nominal":
            value = two_level(value, payload)
    return value


def _pending(rider: Rider, ctx: EventContext) -> Rider:
    """`rider` with the row-space links between where its entry stands and `ctx` appended.

    A `vary` link changes no row space and is left out, as it is everywhere else.
    """
    home = rider.home
    links = () if home is None else row_links(ctx, home)
    return rider if not links else replace(rider, links=(*rider.links, *links))


def row_links(
    ctx: EventContext, below: EventContext | None = None
) -> tuple[tuple[str, str | None], ...]:
    """Return the ROW-SPACE links from `below` (the root by default) down to `ctx`.

    They are the `(kind, label)` pairs a rider carries: a mask has no label, a projection is its
    universe's, and a `vary` link is no row-space link at all. The mask ARRAY is deliberately not
    here — nothing decides from it, and a record is read by people as well as rules.
    """
    root = below
    if root is None:
        root = ctx
        while (parent := parent_of(root)) is not None:
            root = parent
    return tuple(
        (kind, None if kind == "mask" else payload)
        for kind, payload in links_below(ctx, root)
        if kind != "vary"
    )


def _covers(ctx: EventContext, rider: Rider, label: str) -> bool:
    """§2.1/§2.3: whether an OVERLAY replaces the running value at `label`.

    Decided from its RIDER's families against the label's registered point — never from the entry,
    which a re-index leaves a bare member carrying no map.

    Its own universes are its members, placed or not, and it replaces there. Any other label whose
    point names one of its families is replaced too, EXCEPT a universe another family PLACED at a
    point carrying its coordinate: the member declared there is the user's value for that point,
    and a placed universe is exactly one that no coordinate of its own point spells.
    """
    if _named_by(label, rider.families.items()):
        return True
    point = context_session(ctx).systematics.points.get(label)
    if point is None or not any(name in rider.families for name, _tag in point):
        return False
    return _named_by(label, ((name, (tag,)) for name, tag in point))


def _named_by(label: str, coordinates: Iterable[tuple[str, tuple[str, ...]]]) -> bool:
    """Whether one of `coordinates` SPELLS `label`.

    Its own universe `name_tag`, or a joint of one — which is the spelling `_route` gives every
    universe but a placement's.
    """
    return any(
        label == f"{name}_{tag}" or label.startswith(f"{name}_{tag}__")
        for name, tags in coordinates
        for tag in tags
    )


def _entry_state(ctx: EventContext, rider: Rider) -> tuple[bool, str | None]:
    """§2.3, from PROVENANCE alone.

    Whether the ambient at `ctx` still composes this entry, and the universe a projection FIXED it
    into — its own, where every value is that universe.

    An overlay a projection fixed stays fixed through every further row-space link and is never
    re-tested against its re-indexed member, so the universe is the one its RIDER recorded rather
    than any later link; one not yet fixed whose families have no coordinate on the label
    contributes nothing there and is left out.
    """
    if rider.kind != "overlay":
        return True, None
    fixed = rider.fixed_at
    home = rider.home
    for kind, payload in () if home is None else links_below(ctx, home):
        if fixed is not None or kind != "project" or payload == "nominal":
            continue
        if not _covers(ctx, rider, payload):
            return False, None
        fixed = payload
    return True, fixed


def _marked(rider: Rider, fixed_at: str | None) -> Rider:
    """`rider` carrying `_entry_state`'s verdict: the mark and the universe that made it."""
    return replace(rider, fixed=fixed_at is not None, fixed_at=fixed_at)


def _reindex_entry(ctx: EventContext, entry: Any, rider: Rider) -> tuple[Any, str | None]:
    """§2.3's re-index of one ENTRY (never of a user value).

    What the ambient at each universe on the way down READS of it, decided from the rider.

    WHAT becomes of it is `_entry_state`'s answer, read off the rider; this walks the value to
    match. Through a mask, each label's member by that label's mask, as everything else. Through a
    projection to `L`, the TWO-LEVEL member at `L` — a factor computed on shifted objects carries
    its dependence on a shift label one level down, so the one-level read that is right for the
    composed head would silently drop it. A left-out overlay is `None`.
    """
    kept, fixed_at = _entry_state(ctx, rider)
    if not kept:
        return None, None
    home = context_of(entry)
    if home is None or home is ctx:
        return entry, fixed_at
    value = entry
    for kind, payload in links_below(ctx, home):
        if kind == "project" and payload != "nominal":
            value = two_level(value, payload)
        else:
            value = follow(value, kind, payload)
    return with_context(value, ctx), fixed_at


def _expanded(ctx: EventContext) -> tuple[list[Any], list[int], frozenset[int], dict[int, Rider]]:
    """Return the ancestor's live operations re-indexed to `ctx`, order, kinds and slots preserved.

    The expansion a join of an ancestor's factor, or an overlay anchored inside the head, performs.

    Each entry goes through `_reindex_entry`, so an overlay a projection leaves nothing of is left
    out and the rest keep the nominal they had in every row space they came through.
    """
    live, slots, overlays = live_factors(ctx)
    riders = live_riders(ctx)
    entries: list[Any] = []
    kept: list[int] = []
    carried: dict[int, Rider] = {}
    for entry, slot in zip(live, slots, strict=True):
        rider = riders[slot]
        value, fixed_at = _reindex_entry(ctx, entry, rider)
        if value is None:
            continue
        entries.append(value)
        kept.append(slot)
        carried[slot] = _crossed(ctx, rider, value, fixed_at)
    return entries, kept, overlays & frozenset(kept), carried


def _crossed(ctx: EventContext, rider: Rider, value: Any, fixed_at: str | None) -> Rider:
    """Return the rider of an entry an expansion has just re-indexed to `ctx` (§2.3).

    A MASK makes one node two values, so the entry's nominal in the new row space is an identity of
    its own, which a central built there names; a PROJECTION keeps the identity, and the member the
    entry became at the label was recorded when the projection happened.

    Across a non-nominal projection the re-indexed value is that MEMBER, not the entry's nominal —
    for the factor the universe is OF it is the universe itself — so no identity is taken from it:
    reading one would join the owner of that universe, and only once something unrelated had
    re-indexed the entry, deciding one program two ways (§2.3). The member is already recorded.
    """
    home = rider.home
    crossed = () if home is None else links_below(ctx, home)
    priors = rider.priors
    projected = any(kind == "project" and payload != "nominal" for kind, payload in crossed)
    if not projected and any(kind == "mask" for kind, _payload in crossed):
        priors = (*priors, two_level(value, "nominal"))
    return _marked(replace(rider, priors=priors, home=ctx), fixed_at)


def _live_slots(ctx: EventContext) -> tuple[int, ...]:
    """Return the slots of the live PRODUCT factors, in registration order.

    What a read records and what an overlay's prefix is measured against. Overlays are left out:
    one inserted among the factors changes no nominal, so a handle read before it still names the
    same composition (§2.1).
    """
    _live, slots, overlays = live_factors(ctx)
    return tuple(slot for slot in slots if slot not in overlays)


def _prefix_slots(
    ctx: EventContext, slots: Sequence[int], at: int, overlays: frozenset[int], adopted: bool
) -> tuple[int, ...]:
    """Return the FACTOR slots an overlay inserted at position `at` covers.

    The ones it was read over.

    An adopted head stands for the ancestor's whole live list, so an overlay anchored after it
    covers exactly the slots the head stands for, plus whatever this context registered before the
    anchor; once the head is expanded the slots are the list's own.
    """
    held = ambient_of(ctx)
    head = held.head[0] if adopted and held.head is not None else ()
    own = tuple(slot for slot in slots[1 if adopted else 0 : at] if slot not in overlays)
    return (*head, *own)


def ambient_entries(ctx: EventContext) -> tuple[tuple[int, Rider, Any], ...]:
    """Return the live ambient weight as `(slot, rider, entry)` records (§2.3).

    In the order `_compose` applies them — each record's position its index in that order.

    The instrument for what the ambient is MADE of: each operation, its kind, the families it
    carries, the nominal it had in every row space it came through, and the links it was carried
    through to get here. Read from the RIDERS beside the slots, never off the entries, which a
    re-index leaves as bare members carrying neither tags nor role.

    The operations as the ambient at `ctx` COMPOSES them — an overlay at its anchored position,
    before a factor this context registered after it, never the order a lineage walk meets them; an
    overlay a projection leaves nothing of is absent, one a projection fixed into its own universe
    is marked. What a row-space change ADOPTED is one operation per product RUN behind it, because
    the head's node is exactly that product, while each overlay among them keeps its own line: it
    is an operation of its own kind, and a projection into its universe may have fixed it. A
    registration that changes no value leaves the slots, the kinds and the order exactly as they
    were, the joining family entering its slot's families.

    Deliberately not re-exported from `graphed`: it reads the mechanism, and the analysis idiom is
    `graphed.weight` / `graphed.variations`.
    """
    if not isinstance(ctx, ContextNode):
        raise GraphedError(
            "graphed.context.ambient_entries reads an event context's ambient weight operations"
        )
    live, slots, _overlays = live_factors(ctx)
    riders = live_riders(ctx)
    own = frozenset(ambient_of(ctx).slots)
    composed: list[tuple[int, Rider, Any]] = []
    run: list[int] = []
    for entry, slot in zip(live, slots, strict=True):
        rider = riders[slot]
        kept, fixed_at = _entry_state(ctx, rider)
        if not kept:
            continue
        if slot not in own and rider.kind != "overlay":
            run.append(slot)
            continue
        if run:
            composed.append(_adopted_record(ctx, run, riders))
            run = []
        composed.append((slot, _marked(rider, fixed_at), entry))
    if run:
        composed.append(_adopted_record(ctx, run, riders))
    return tuple(composed)


def _adopted_record(
    ctx: EventContext, run: Sequence[int], riders: Mapping[int, Rider]
) -> tuple[int, Rider, Any]:
    """One product RUN behind an adopted head, as the single operation the head's node is.

    Its families are every family the operations in the run carry, a join of one of them included,
    and its links are the ones the lineage took to get here — the head came through all of them.
    """
    held = ambient_of(ctx)
    families = {name: tags for slot in run for name, tags in riders[slot].families.items()}
    return (
        held.slots[0],
        Rider("factor", families, home=ctx, links=row_links(ctx)),
        held.factors[0],
    )


def _lineage_reads(ctx: EventContext) -> Iterator[tuple[EventContext, _Read]]:
    """Every read record `ctx` can name — its own and every ancestor row space's.

    Unlike `live_factors`, this does NOT stop where the adoption ended: a family that joined an
    ancestor's factor expands the head into that ancestor's operations re-indexed here, keeping
    their slots and generations, so a handle read before the change still names a prefix of the
    live factor slots and is matched like any other record (§2.3). The records are keyed by the
    ancestor's member nodes, which is what a crossed handle carries.

    Each record comes with the context whose row space it was recorded in, because a handle carried
    across a cut carries that row space's nodes re-indexed, not the nodes the record holds.

    A `vary` link shares the list object, so a list already yielded is skipped.
    """
    node: EventContext | None = ctx
    seen: set[int] = set()
    while node is not None:
        if id(ambient_of(node).reads) not in seen:
            seen.add(id(ambient_of(node).reads))
            for record in ambient_of(node).reads:
                yield node, record
        node = parent_of(node)


def _carried_up(
    ctx: EventContext, home: EventContext, central: Any
) -> tuple[tuple[int, ...], ...]:
    """Return the node tuples `central` can present to a record in `home`'s row space.

    §2.3.

    A handle read at an ancestor names that read's composition in either spelling: kept as it was
    read, where its members ARE the record's nodes, or carried down with `graphed.reindex_to`,
    which re-indexes every member across the intervening cuts. The second is undone here, one
    mask-follow per link off the Session's op record, per label, which is what lets a CROSSED
    handle be matched — and found stale — against the record the composition it names actually
    wrote. Only the first is offered when some member is not that re-index, which is a handle
    naming no composition below.
    """
    own = member_nodes(central)
    masks = tuple(payload for kind, payload in links_below(ctx, home) if kind == "mask")
    if not masks:
        return (own,)
    session = context_session(ctx)

    def undo(node: int, label: str) -> int | None:
        for mask in reversed(masks):
            recorded = session.recorded_op(node)
            carried = member_of(mask, label)
            if (
                recorded is None
                or recorded[0] != "getitem"
                or recorded[2][1:] != (carried.node_id,)
            ):
                return None
            node = recorded[2][0]
        return node

    members = members_of(central).items()
    lifted: list[int] = []
    for label, member in members:
        for node in member_nodes(member):
            one = undo(node, label)
            if one is None:
                return (own,)
            lifted.append(one)
    return (own, tuple(lifted))


def _extension(ctx: EventContext, central: Any) -> tuple[str, Any] | None:
    """§2.1: what a weight registration's central NAMES.

    A composition a read handed out (`("ambient", the index the overlay goes at)`, or
    `("prefix", the slot it was read over)` when that index is inside an adopted head the
    registration must expand first), a composition a later join has since WIDENED
    (`("stale", the families that grew)`, which `weight_form` refuses), an already-registered
    factor (`("factor", its slot)`), or nothing new.

    Three answers are refusals `weight_form` raises on (§2.3): `("owned", the universe)` for a
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
    held = ambient_of(ctx)
    node = member_of(central, "nominal")  # ONE level in, not `two_level`, which peels a second
    if isinstance(node, Varied):  # nested past §2.2's one level; `_check_forms` names it properly
        return None
    _live, slots, overlays = live_factors(ctx)
    riders = live_riders(ctx)
    # a composition a READ recorded, WITHOUT composing anything here: its FACTORS must still be a
    # prefix of the live factors, which is what makes the handle a rescaling of the operations it
    # was built from and lets the overlay land right after them. Reads are recorded per row space,
    # so the member nodes alone answer which composition the handle is.
    factors = tuple(slot for slot in slots if slot not in overlays)
    keys: dict[int, tuple[tuple[int, ...], ...]] = {}
    stale: list[int] = []
    for home, (members, read, stamps) in _lineage_reads(ctx):
        if id(home) not in keys:
            keys[id(home)] = _carried_up(ctx, home, central)
        if members not in keys[id(home)] or read != factors[: len(read)]:
            continue
        moved = [
            slot for slot, was in zip(read, stamps, strict=True) if held.gens.get(slot, 0) != was
        ]
        if moved:  # a join has since ADDED universes under this handle (§2.1)
            stale = moved
            continue
        at = _overlay_index(ctx, read)
        if at is not None and read[-1] not in held.slots:
            # the record is the adopted head's own tuple: the overlay anchors right after it and
            # nothing is re-indexed, so no operation lands inside a covered prefix (§2.3)
            return ("ambient", at)
        if not read:
            continue
        covered = _fixed_cover(ctx, read[-1], slots, riders)
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
    # factor's place; only a product factor names a factor. The ENTRIES are not read at all here:
    # every candidate comes from the rider beside the slot (§3's provenance principle).
    # §2.3: the NOMINAL arm answers FIRST over every live slot, then the projected member — one
    # node can be both a live factor's nominal in the central's own row space and another entry's
    # recorded member, and there the join keeps the universe projected into where the refusal would
    # not. The arms are the outer loop, so registration order cannot pick the answer.
    for arm in (_names_nominal, _names_member):
        for slot in slots:
            if slot in overlays:
                continue
            match = arm(ctx, riders[slot], node, slot)
            if match is None:
                continue
            if match[0] == "factor":
                covered = _fixed_cover(ctx, slot, slots, riders)
                if covered is not None:
                    return ("covered", covered)
            return match
    return None


def _names_nominal(
    ctx: EventContext, rider: Rider, node: Any, slot: int
) -> tuple[str, Any] | None:
    """§2.3: whether a central names this entry's NOMINAL identity in the central's own row space.

    The entry itself (`("factor", slot)`), or the universe this context is projected into that the
    entry OWNS (`("owned", L)`, which `weight_form` refuses).

    The candidates are the rider's nodes only, never the entry as it currently stands: at a
    projection that object's nominal is its MEMBER at the label, which for the factor the universe
    is OF is the universe itself — so reading it would join the owner, and only once something
    unrelated had re-indexed the entry, deciding one program two ways.
    """
    for candidate in rider.priors:
        if _same_node(candidate, node):
            owned = _owned_projection(ctx, candidate, rider)
            return ("owned", owned) if owned is not None else ("factor", slot)
    return None


def _names_member(ctx: EventContext, rider: Rider, node: Any, slot: int) -> tuple[str, Any] | None:
    """§2.3: whether a central names the member this entry BECAME at a projected label.

    The entry itself (`("factor", slot)`) when it owns no universe the context is still inside, the
    refusal (`("owned", L)`) when it owns one.

    The member matched is the INNERMOST record (`_member_here`), while ownership is tested against
    EVERY universe the context is still inside: below stacked projections the owner of an outer
    universe named by its recorded member is refused exactly as it is when named by its nominal,
    and a join there would erase the outer universe.

    A recorded member is matched HOWEVER the central was built — at the projection, at the parent,
    or after an unrelated expansion — and a join is the answer when the entry owns
    nothing: inside `jes_up` the SF re-derived over this context's shifted jets IS the
    jes-dependent factor's member there, and a new factor would square it.
    """
    recorded = _member_here(rider)
    if recorded is None or not _same_node(recorded[1], node):
        return None
    owned = _owned_inside(ctx, rider)
    return ("owned", owned) if owned is not None else ("factor", slot)


def _member_here(rider: Rider) -> tuple[str, Any] | None:
    """`(label, member)`: what this entry IS where it now stands.

    The member recorded at the INNERMOST universe its history was projected into, read off the
    rider.

    An OUTER projection's record describes a row space the context has since left: below a second
    projection the entry is its member there, and naming the outer record would join a value that
    is no longer the entry's. `None` for an entry no projection has carried.
    """
    label = next(
        (
            payload
            for kind, payload in reversed(rider.links)
            if kind == "project" and payload != "nominal"
        ),
        None,
    )
    if label is None or label not in rider.projected:
        return None
    return label, rider.projected[label]


def _owned_projection(ctx: EventContext, candidate: Any, rider: Rider) -> str | None:
    """§2.3: the universe between the candidate's row space and `ctx` this entry OWNS.

    The entry owns the universe projected into there when a family of its RIDER is a coordinate of
    that universe's point; else `None`.

    A projection to another universe keeps the entries' identity (it re-indexes the composed
    product, not the nodes a central names), which is what lets a central built above it name its
    factor there. For the factor the universe is OF, that is a trap: its member at the label IS the
    universe projected into, so joining would put the central back and erase it. A joint label is
    owned by every family in its point, which is why the point decides and not the label's name —
    and the families come from the rider, because the re-index the expansion performs leaves a bare
    member behind, with no tag map for this to read.
    """
    home = context_of(candidate)
    if home is None or not is_ancestor(home, ctx):
        return None
    return next(
        (
            payload
            for kind, payload in links_below(ctx, home)
            if kind == "project"
            and payload is not None
            and payload != "nominal"
            and _owns(ctx, rider, payload)
        ),
        None,
    )


def _owned_inside(ctx: EventContext, rider: Rider) -> str | None:
    """§2.3: the first universe this entry OWNS among ALL the ones the context is still inside.

    The non-nominal universes its history was projected into, outermost first — or `None`.

    Below stacked projections the context is inside every one of them, so an entry named by its
    recorded member at the innermost label is still the owner of an outer universe and refused
    there, as it is when named by its nominal identity.
    """
    return next(
        (
            payload
            for kind, payload in rider.links
            if kind == "project"
            and payload is not None
            and payload != "nominal"
            and _owns(ctx, rider, payload)
        ),
        None,
    )


def _owns(ctx: EventContext, rider: Rider, label: str) -> bool:
    """§2.3: whether this entry is the factor `label` is OF.

    A family of its RIDER is a coordinate of that universe's point, so a joint or a placed label
    has several owners and the point decides rather than the label's name.
    """
    points = context_session(ctx).systematics.points
    return any(name in rider.families for name, _tag in points.get(label, ()))


def _fixed_cover(
    ctx: EventContext,
    slot: int,
    slots: Sequence[int],
    riders: Mapping[int, Rider],
) -> tuple[str, str] | None:
    """§2.3: the `(family, universe)` of an overlay that covers `slot` and is FIXED at `ctx`.

    Projected into its own universe, where its member is the whole ambient rescaled and every value
    is that universe.

    An operation landing inside the factors such an overlay covers has nothing to re-derive it
    over: the overlay's member is the user's node over the OLD product. `None` when no overlay
    covers the slot, or when the one that does still composes as an overlay.

    The universe is the one the overlay's RIDER recorded, not the last link this context took: a
    further projection below it adds links that fixed nothing, and naming one of those would tell
    the user the wrong universe to read the handle at.
    """
    for entry_slot in slots:
        rider = riders.get(entry_slot)
        if rider is None or slot not in rider.prefix:
            continue
        kept, fixed_at = _entry_state(ctx, rider)
        if kept and fixed_at is not None:
            return next(iter(rider.families), rider.kind), fixed_at
    return None


def _overlay_index(ctx: EventContext, read: tuple[int, ...]) -> int | None:
    """Where in `ctx`'s OWN entry list an overlay over `read` belongs.

    Right after the last FACTOR it was composed over and after any overlay already anchored there,
    so two relative-delta families over the same factors stay in registration order. `None` when
    that factor is not this list's to order (a prefix ending inside an ancestor's entries, which no
    read here records).
    """
    held = ambient_of(ctx)
    if not read:
        return None
    if read[-1] in held.slots:
        at = held.slots.index(read[-1]) + 1
    elif held.adopted and held.head is not None and read == held.head[0]:
        # the adopted head stands for exactly this prefix, so the overlay goes after it and the
        # ancestor's operations stay folded inside it — nothing is re-indexed
        at = 1
    else:
        return None
    while at < len(held.slots) and held.slots[at] in held.overlays:
        at += 1
    return at


def _same_node(left: Any, right: Any) -> bool:
    """Whether a central names an entry's RECORDED node.

    Its nominal identity, or the member it became at a projected label: one node, one row space,
    read as one value (§2.3).

    The ids answer for almost every pair. A MASK between the two handles makes one id two different
    values, so it splits them — unless the central IS the recorded node carried down through that
    mask, which `_reindexed_onto` reads off the Session's op record. A `vary` link, a projection to
    `nominal` and a projection to any other universe all keep the row space and the identity (a
    projection re-indexes the composed product, not the nodes a central names), so a central built
    above a projection, or re-stamped at it, names its factor there as at the parent.
    """
    return _one_value(left, right) or _reindexed_onto(left, right)


def _one_value(left: Any, right: Any) -> bool:
    """Whether two values are the same IR node read as the same VALUE, along ONE lineage.

    Contexts off one lineage never match: a branch that diverged carries its own rows. Nothing more
    is asked of the link chain between them, because crossing a CUT mints: every member is a
    `getitem` of its own, so one node id never stands in two row spaces separated by a mask
    (measured — `reindex_to` across a cut answers with the re-indexed node, a re-derivation with a
    new one, and a value the user simply carries down keeps the row space it was built in). A
    central that IS the recorded node carried down is matched by `_reindexed_onto` instead, off the
    op record.
    """
    if left.node_id != right.node_id:
        return False
    here, there = context_of(left), context_of(right)
    if here is None or there is None or here is there:
        return True
    deep, shallow = (here, there) if is_ancestor(there, here) else (there, here)
    return bool(is_ancestor(shallow, deep))


def _reindexed_onto(recorded: Any, node: Any) -> bool:
    """§2.3: whether `node` is `recorded` RE-INDEXED into `node`'s own row space.

    One mask-follow per link between the two contexts, peeled off the Session's op record.

    A central built at an ancestor and carried down with `graphed.reindex_to` is the very node the
    expansion re-indexes the entry to (interning makes the two one node), so it names that entry as
    a central built at the ancestor does. DECIDING still mints nothing: the re-index the user
    already performed is what recorded the ops this reads, and an expression merely re-derived
    below the mask carries no such record — which is what keeps §4's re-derivation controls new
    factors.
    """
    here, there = context_of(recorded), context_of(node)
    if here is None or there is None or not is_ancestor(here, there):
        return False
    session = node.session
    current = node.node_id
    for kind, payload in reversed(links_below(there, here)):
        if kind != "mask":
            continue
        entry = session.recorded_op(current)
        mask = member_of(payload, "nominal")
        if entry is None or entry[0] != "getitem" or entry[2][1:] != (mask.node_id,):
            return False
        current = entry[2][0]
    return int(current) == int(recorded.node_id)


#: what an adopted head STANDS FOR here: the operations it composes, their slots and the overlays
#: among them — re-indexed into this row space once a join has extended one of them (§2.3)
_Stood = tuple[list[Any], list[int], frozenset[int]]


def _live_slots_behind(ctx: EventContext, live: Sequence[int]) -> tuple[int, ...]:
    """Return the slots the adopted head stands for, in the live order.

    The ones a join re-composes.
    """
    standing = _stood_for(ctx)
    return tuple(live) if standing is None else tuple(standing[1])


def _extend(
    ctx: EventContext,
    slot: int,
    members: Mapping[str, Any],
    name: str,
    family: tuple[str, ...],
) -> tuple[Varied, list[Any], list[int], frozenset[int], bool, dict[int, Rider], _Stood | None]:
    """§2.1's joins-a-factor outcome, multiplying the named container in exactly ONCE.

    The outcome is
    `(the container the family joined, the new entry list, its slots, the overlay slots in it,
    whether the union WIDENED the joined nominal member, the riders, what an adopted head now
    stands for)`, written so the named container is multiplied in exactly ONCE.

    A factor of an ANCESTOR is reached through the composed container this context ADOPTED, which
    keeps its slot and its node — it IS the composition the row-space change selected, and §2.3's
    join keeps the universe projected into — and gains one universe per label the join changed,
    each the product of the operations the head stands for with the joined one's member in place.
    That is the one place that pays §2.1(b)'s factors x labels, and only for the labels that moved.
    The joined container keeps its SLOT, so order and kind are preserved: a family joining a factor
    registered before an overlay keeps that factor's position and the overlay still replaces the
    product of the operations it was read over.
    """
    held = ambient_of(ctx)
    covering = _covering_overlay(ctx, slot)
    if slot in held.slots:
        at = held.slots.index(slot)
        joined, widened = _joined(held.factors[at], members, name, family, ctx, covering)
        entries = list(held.factors)
        entries[at] = joined
        # a join keeps the nominal node, so the entry stands for what it always did
        riders = {**held.riders, slot: _carries(held.riders[slot], name, family)}
        return joined, entries, list(held.slots), held.overlays, widened, riders, None
    live, live_slots, live_overlays, riders = _expanded(ctx)
    # only the operations the HEAD stands for are re-composed: the ones this context registered
    # AFTER the adoption keep their own places in the list behind it, and composing them into the
    # head as well would multiply each of them twice at the labels the join moved.
    behind = frozenset(_live_slots_behind(ctx, live_slots))
    stood = [
        entry for entry, entry_slot in zip(live, live_slots, strict=True) if entry_slot in behind
    ]
    slots = [entry_slot for entry_slot in live_slots if entry_slot in behind]
    overlays = live_overlays & behind
    at = slots.index(slot)
    standing = stood[at]
    joined, widened = _joined(standing, members, name, family, ctx, covering)
    stood[at] = joined
    riders[slot] = _carries(riders[slot], name, family)
    fixed = frozenset(entry_slot for entry_slot, rider in riders.items() if rider.fixed)
    head = _headed(ctx, stood, slots, overlays, fixed, standing, joined, name, family)
    return (
        joined,
        [head, *held.factors[1:]],
        list(held.slots),
        held.overlays,
        widened,
        riders,
        (
            stood,
            slots,
            overlays,
        ),
    )


def _headed(
    ctx: EventContext,
    stood: Sequence[Any],
    slots: Sequence[int],
    overlays: frozenset[int],
    fixed: frozenset[int],
    standing: Any,
    joined: Any,
    name: str,
    family: tuple[str, ...],
) -> Varied:
    """Return the adopted head carrying the universes a join just added.

    The join is of an operation the head stands for.

    Its node is kept at every label it already answers — the head IS the composition this context
    adopted, and re-composing the operations behind it would answer the projection's own universe
    with an equal value under a new node. A label the join MOVED (the family's own universes, its
    joints, a coordinate the union added — never a label the joined operation contributes its
    nominal at, which the head already answers) becomes the product of the operations the head
    stands for, read at that label with the joined member in place.
    """
    current = ambient_of(ctx).factors[0]
    carried = dict(members_of(current))
    moved = [
        label
        for label in _union(context_labels(ctx), labels_of(joined))
        if label != "nominal"
        and label not in carried
        and member_nodes(two_level(joined, label)) != member_nodes(two_level(standing, label))
    ]
    products = _compose(
        stood,
        moved,
        overlays=_marks(stood, slots, overlays),
        fixed=_marks(stood, slots, fixed),
    )
    tags = {**tags_of(current), name: family}
    return rebuild({**carried, **products}, tags=tags, context=ctx)


def _carries(rider: Rider, name: str, family: tuple[str, ...]) -> Rider:
    """§2.3: a join EXTENDS the rider's families.

    The joined container carries the new one too, and every rule that reads them (ownership, an
    overlay's cover, a widening's message) must see it. The nominal node is untouched by a join, so
    the identity chain is unchanged.
    """
    return replace(rider, families={**rider.families, name: family})


def _covering_overlay(ctx: EventContext, slot: int) -> str | None:
    """Return the family of the first OVERLAY whose prefix covers `slot`, `None` when none does."""
    _live, slots, _overlays = live_factors(ctx)
    riders = live_riders(ctx)
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


def _marks(
    entries: Sequence[Any], slots: Sequence[int], overlays: frozenset[int]
) -> frozenset[int]:
    """Return the overlay slots as the OBJECT ids `_compose` tests.

    See `_materialise`.
    """
    return frozenset(
        id(entry) for entry, slot in zip(entries, slots, strict=True) if slot in overlays
    )


def _overlay(
    ctx: EventContext, members: Mapping[str, Any], name: str, family: tuple[str, ...]
) -> Varied:
    """§2.1's relative-delta family as an ORDERED operation on the ambient.

    Its NOMINAL is the ambient's own nominal node, which is what makes it an overlay rather than a
    factor: the composition REPLACES the running value with its member at the family's labels
    instead of multiplying, because that member already IS the whole ambient rescaled. Every other
    label reads its nominal, so it contributes nothing there and the factors compose as before.
    """
    return rebuild(
        {**members, "nominal": two_level(members["nominal"], "nominal")},
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
    """Return `container` with the joining family's universes among its own members.

    Over the UNION of the two centrals' coordinates (§2.1).

    The shared node stays the nominal, and every coordinate either central declares — a central
    computed over shifted objects IS the weight's dependence on that shift — is carried, so the
    ambient reads the weight at each label's own objects whichever family registered first. The
    same coordinate declared by both with different nodes is two weights, not one, and is refused
    here, before anything of this registration is recorded. A projection's adopted member is a bare
    array and becomes the nominal.
    """
    existing = members_of(container)
    added = {label: member for label, member in members.items() if label != "nominal"}
    tags = {**tags_of(container), name: family}
    nominal, widened = _union_nominal(existing["nominal"], members["nominal"], name, ctx, covering)
    return rebuild({**existing, "nominal": nominal, **added}, tags=tags, context=ctx), widened


def _check_widening(ctx: EventContext, slot: int, central: Any, name: str) -> None:
    """§2.1's refusal of a join that would ADD universes under an OVERLAY.

    The join would add them to a nominal member an OVERLAY was read over: the overlay's members ARE
    the composition as it stood when its handle was read, and widening a factor under it would
    change that composition beneath them.

    Decided from LABEL SETS, both operands read at the level THIS context reads them: a mask
    carries a label set unchanged but a projection RESETS it, so a central built above one joins as
    its member there and comparing it at its own depth would refuse a join whose union adds nothing
    — §2.1's "a central built at an ancestor still names its factor after the expansion". Nothing
    that mints is walked, and the answer lands before `gather_members` mints this family's cross
    members, so the refusal leaves not one node behind (§2.5), wherever the factor it names lives.
    """
    live, slots, _overlays = live_factors(ctx)
    entry = _here(ctx, live[slots.index(slot)])
    added = _coordinates(_here(ctx, central)) - _coordinates(member_of(entry, "nominal"))
    covering = _covering_overlay(ctx, slot) if added else None
    if covering is None:
        return
    raise GraphedError(
        f"graphed.vary({name!r}): its central names the weight factor registered here, but "
        f"joining would add the universes {sorted(added)} to it, and the "
        f"relative-delta family {covering!r} was registered on a graphed.weight() handle read "
        "over that factor; register the absolute family first and the relative-delta family "
        "on a handle read after it"
    )


def _here(ctx: EventContext, value: Any) -> Any:
    """Return `value` at the level `ctx` reads it, peeled through the PROJECTIONS.

    Only the projections between its row space and this one, and nothing else — the label-set half
    of the re-index `weight_form` performs two statements on (`handles.follow`). A mask leaves
    the set alone and is not followed: following one mints, and this decides a refusal that must
    leave no node behind (§2.5).
    """
    home = context_of(value)
    if home is None or home is ctx:
        return value
    for kind, payload in links_below(ctx, home):
        if kind == "project":
            value = member_of(value, payload)
    return value


def _coordinates(value: Any) -> frozenset[str]:
    """Return the universes one operand of the union declares.

    `{"nominal"}` for a bare member, which is what `_union_nominal` reads it as.
    """
    return frozenset(members_of(value))


def _union_nominal(
    carried: Any, central: Any, name: str, ctx: EventContext, covering: str | None
) -> tuple[Any, bool]:
    """Return the joined container's NOMINAL.

    The shared node, carrying every coordinate either central declares. A central computed over
    shifted objects is a container whose universes ARE the weight's dependence on that shift, and
    `two_level` reads them at the ambient's shift labels, so the union is what makes the ambient
    read the weight at each label's own objects whichever family registered first.
    """
    left, right = members_of(carried), members_of(central)
    for label, member in right.items():
        if (
            label != "nominal"
            and label in left
            and member_nodes(left[label]) != member_nodes(member)
        ):
            raise GraphedError(
                f"graphed.vary({name!r}): its central names the weight factor already registered "
                f"here, but the two disagree at coordinate {label!r} — the registered central "
                f"carries node {member_nodes(left[label])} there and this one carries "
                f"{member_nodes(member)}; one weight cannot have two values at one coordinate"
            )
    merged = {**left, **right}
    widened = set(merged) > set(left)
    if len(merged) == 1:
        return merged["nominal"], widened
    tags = {**tags_of(carried), **tags_of(central)}
    return rebuild(merged, tags=tags, context=ctx), widened


def weight_form(
    ctx: EventContext,
    name: str,
    central: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None,
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
    tags: Mapping[str, Any],
) -> tuple[AmbientRecord, list[Any] | None]:
    """Overload (b): register a per-event weight factor into the context's ambient weight.

    Returns the record of the `vary` child the registration derives, and the operands its cache
    folds onto (`fold`) when the factor appends to a foldable memo, else `None`. The factor is
    composed label-aligned per §2.4 with whatever is already registered.
    """
    held = ambient_of(ctx)
    if collections is not None:
        raise GraphedError("collections= belongs to the shift form; a weight form takes tags")
    if central is None:
        raise GraphedError(
            f"the weight form of graphed.vary({name!r}) needs the central per-event factor as its "
            "third positional argument"
        )
    ambient_tags = _ambient_tags(ctx)
    inherited = held.weight_tags.get(name, ())
    # What the next composition will multiply, decided HERE, before this registration mints, and by
    # the same predicate the read uses: onto a foldable memo the new factor folds (today's
    # two-element chain step, so a program reading at every intermediate context pays today's node
    # count and no more); otherwise the composition is remade from the original factors and this
    # registration joins them.
    base = _ambient_operands(ctx)
    folds = _foldable(ctx) is not None
    # §2.1: the central NAMES the factor it varies. Decided here, before anything is minted, and
    # read only for its answer — nothing is composed or re-indexed to reach it, so a program that
    # names nothing records exactly the nodes it records without this.
    match = _extension(ctx, central)
    if match is not None and match[0] == "stale":
        # §2.1: the handle names a composition a later join WIDENED, so its universes are not that
        # composition's any more. Refused here, before this registration mints anything at all.
        raise GraphedError(
            f"graphed.vary({name!r}): its central is a graphed.weight() handle read BEFORE a "
            f"later registration added universes to the weight factor it composed ({match[1]}), "
            "so the handle's universes are no longer that composition; read the handle again "
            "after that registration (`w = graphed.weight(ctx)`) and register this family on the "
            "new handle"
        )
    if match is not None and match[0] == "owned":
        # §2.3: this context IS that universe, and the named factor's member there is what makes it
        # one — joining would put the central back and the projection would vanish
        raise GraphedError(
            f"graphed.vary({name!r}): its central names the weight factor that the universe "
            f"{match[1]!r} this context is projected into is OF, whose member there is that "
            "universe rather than the central; register this family on the factor at the parent, "
            "before the projection, or read a graphed.weight() handle AT this context "
            "(`w = graphed.weight(ctx)`) and register this family on that"
        )
    if match is not None and match[0] == "covered":
        # §2.3: inside the overlay's own universe its member is the whole ambient rescaled — the
        # user's node over the OLD product — and nothing is minted to re-derive it over a new one
        covered_family, universe_label = match[1]
        raise GraphedError(
            f"graphed.vary({name!r}): its central names a weight factor that the relative-delta "
            f"family {covered_family!r} was registered over, and this context is projected into "
            f"{universe_label!r}, that family's own universe, where its member IS the whole "
            "ambient and joining underneath it would change the composition its members already "
            "are; register this family on the factor at the parent, before the projection, or "
            "read a graphed.weight() handle AT this context and register the family on that"
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
    if match is not None and match[0] == "factor":
        _check_widening(ctx, match[1], central, name)
    # The families the live ambient's RIDERS carry are the CANDIDATES for composition (m56): a
    # member's coordinate on one of them is dropped iff the member's node at that label reads a
    # lineage factor's varied member there (`_reads_ambient` in `vary._foreign`), which the
    # composition below would multiply in again via `two_level`; reached through shifted objects
    # instead, it fans out. The riders rather than the containers' tag maps, because a row-space
    # change hands its child ONE composed member carrying no map: the families it stands for are
    # still composed here (a projection answers them with the universe it projected into), so a
    # crossed handle's member is resolved by this ambient and not a cross-term to fan out. The set
    # is over-inclusive — a mask-derived child's adopted container carries leaked shifts — and that
    # is harmless because the node test decides.
    composed = frozenset(name for rider in live_riders(ctx).values() for name in rider.families)
    one_at_a_time, joints = gather_members(
        name,
        tags,
        variations,
        inherited,
        points,
        session=context_session(ctx),
        carriers=_carriers(ctx),
        composes_as_union=composes_as_union,
        max_universes=max_universes,
        composed=composed,
    )
    # §1.1's within-the-container clause, keyed by family NAME over the three carriers
    # `context_labels` reads. `_members` alone is the §2.4 union and cannot say which family a
    # label came from; the same `name` is the correlated case (one knob, §2.1) and is admitted —
    # `check_family` refuses a repeated tag within it — while any OTHER family already spelling
    # this label would make one universe differ from nominal in two knobs.
    registered = {
        f"{n}_{t}"
        for tag_map in (
            ambient_tags,
            *(tags_of(source) for source in _carriers(ctx)[1:]),
        )
        for n, ts in tag_map.items()
        if n != name
        for t in ts
    }
    for label in one_at_a_time:
        if label in registered:
            raise GraphedError(f"variation label {label!r} is already carried by this container")
    # a machine-minted joint joins the factor container as a flat cross node so `two_level` reads
    # it by name, but it is a cross-coordinate, not a tag of this family, so it stays out of
    # `_tags`.
    factors = {"nominal": central, **one_at_a_time, **joints}
    check_members(factors)
    # §2.1(b)'s ROW-SPACE rule: an ancestor-handled factor is re-indexed across the intervening
    # links; a descendant or divergent one is a construction-time error naming the direction.
    factors = {label: reindex_to(factor, ctx) for label, factor in factors.items()}
    family = inherited + _tags_of(name, one_at_a_time)
    if match is not None and match[0] == "factor":
        # joining an existing factor REMAKES the composition from the new entry list: the fold memo
        # is built on the premise that a registration only ever appends
        factor, updated, slots, overlays, widened, riders, stood = _extend(
            ctx, match[1], factors, name, family
        )
        joined_slot: int | None = match[1] if widened else None
        operands = updated
    else:
        joined_slot = stood = None
        slot = next(_SLOT)
        entries, entry_slots, riders = held.factors, held.slots, held.riders
        adopted = held.adopted
        prefix: tuple[int, ...] = ()
        if match is None:
            factor = rebuild(factors, tags={name: family}, context=ctx)
            at, overlays, kind = len(held.factors), held.overlays, "factor"
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
                at, overlays = match[1], held.overlays
            prefix = _prefix_slots(ctx, entry_slots, at, overlays, adopted)
            overlays = overlays | {slot}
        riders = {
            **riders,
            slot: Rider(
                kind,
                {name: family},
                priors=(two_level(factor, "nominal"),),
                prefix=prefix,
                home=ctx,
                links=row_links(ctx),
            ),
        }
        updated = [*entries[:at], factor, *entries[at:]]
        slots = [*entry_slots[:at], slot, *entry_slots[at:]]
        # an entry at the END is an append the memo folds onto — the overlay replaces the folded
        # composition at its own labels, which is what its members already are
        operands = [*base, factor] if at == len(entries) else updated
    # the §2.4 union is RECORDED here, never recomputed at the read: recomputing it from the
    # context would hand the ambient every shift registered after it, and recomputing it from the
    # factors would drop the shift labels a factor computed on shifted objects is read through.
    recorded = _union(context_labels(ctx), tuple(factors))
    if operands is not updated and not _resolvable(base, tuple(factors)):
        operands = updated  # the memo cannot answer a label this call minted (below)
    # the record-time type check, run BEFORE anything is recorded so a refused registration leaves
    # no trace at all
    _check_forms(
        context_session(ctx),
        operands,
        recorded,
        _marks(updated, slots, overlays),
        _marks(updated, slots, frozenset(s for s, rider in riders.items() if rider.fixed)),
    )
    names: tuple[str, ...] = ()
    if match is None:
        form = "factor"
    elif match[0] == "factor":
        form = "join"
        names = tuple(sorted(set(riders[match[1]].families) - {name}))
    else:
        form = "overlay"
        names = tuple(
            sorted(
                {
                    carried
                    for covered_slot in prefix
                    for carried in (
                        riders[covered_slot].families if covered_slot in riders else ()
                    )
                }
            )
        )
    record_labels(factor)  # §2.5's vary-time half; the members are stamped when they compose
    # §2.5's shift-after-weight operand one: this factor's OWN member node ids, by value.
    context_session(ctx).systematics.record_weight_factor(name, member_nodes(factor))

    record = for_vary_link(ctx)
    record.registration = Registration(name, Kind.WEIGHT, family, ctx, form, names)
    record.weight_tags[name] = family
    record.factors = updated
    record.slots = slots
    record.overlays = overlays
    record.riders = riders
    if joined_slot is not None:
        # §2.1's staleness stamp: every handle read over this factor before now composed universes
        # this join has just added to, and naming one of them is refused from here on
        record.gens[joined_slot] = held.gens.get(joined_slot, 0) + 1
    # the adoption marker outlives a registration only while the head still STANDS FOR the parent's
    # live operations: a join extends it and it does (`stood` is what it stands for here, the
    # operations re-indexed with that join applied); an overlay anchored inside it replaced it with
    # them and it does not
    record.adopted = held.adopted and (stood is not None or updated[0] is held.factors[0])
    record.head_ops = stood if record.adopted else None
    record.recorded = recorded
    record.origin = None
    record.memo = None
    # a join, and an overlay inserted before the end, remake the composition from the entry list;
    # only an append composed onto the memo above can fold onto it (§3)
    return record, (operands if folds and operands is not updated else None)


def shift_form(
    ctx: EventContext,
    name: str,
    nominal: object,
    variations: Mapping[Any, Any] | None,
    collections: Mapping[str, Mapping[Any, Any] | Varied] | None,
    points: Iterable[Mapping[str, Any]] | None,
    composes_as_union: bool,
    max_universes: int,
    tags: Mapping[str, Any],
) -> tuple[dict[str, Any], AmbientRecord]:
    """Overload (c): replace each named collection with a `Varied` over one shared tag set.

    Returns the collections the call varies, by name, and the record of the `vary` child.
    """
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
        raise GraphedError(
            f"the shift form of graphed.vary({name!r}) needs at least one collection"
        )
    for collection_name, inner in list(mapping.items()):
        if isinstance(inner, Varied):  # m55: lockstep by propagation, unpacked to the hand form
            mapping[collection_name] = _unpack_varied(ctx, name, collection_name, inner, points)
    _check_lockstep(name, mapping)

    replaced: dict[str, Any] = {}
    family: tuple[str, ...] = ()
    for collection_name, inner in mapping.items():
        current = ctx[collection_name]
        inherited = tags_of(current).get(name, ())
        one_at_a_time, joints = gather_members(
            name,
            inner,
            None,
            inherited,
            points,
            session=context_session(ctx),
            carriers=_carriers(ctx),
            composes_as_union=composes_as_union,
            max_universes=max_universes,
        )
        existing = dict(members_of(current))
        # §4.6: a supplied member is projected by the label's own point, not flattened to its
        # central universe — which is what makes a shift (x) shift joint point expressible; a
        # machine-minted joint is already the cross node, so `member_of` on it is the identity
        resolved = {
            label: member_of(member, label)
            for label, member in {**one_at_a_time, **joints}.items()
        }
        check_members({**existing, **resolved})
        resolved = {label: reindex_to(member, ctx) for label, member in resolved.items()}
        for label in resolved:
            if label in existing:
                raise GraphedError(f"variation label {label!r} already varies {collection_name!r}")
        tag_map = dict(tags_of(current))
        # a joint is a cross-coordinate, not a tag of this single family, so it stays out of
        # `_tags`
        family = tag_map[name] = inherited + _tags_of(name, one_at_a_time)
        replaced[collection_name] = register(
            rebuild({**existing, **resolved}, tags=tag_map, context=ctx)
        )
        _report_shift_after_weight(ctx, collection_name, existing)
    record = for_vary_link(ctx)
    record.registration = Registration(
        name, Kind.SHIFT, family, ctx, "shift", varies=tuple(mapping)
    )
    return replaced, record


def _report_shift_after_weight(
    ctx: EventContext, collection: str, pre_shift: Mapping[str, Any]
) -> None:
    """§2.5/§2.1: report weight factors registered BEFORE the collection they read varies.

    Such a factor fills every shift universe with its PRE-shift value, and the registry is not
    re-derived, so it is unfixable after the fact. Report each such family paired with the
    collection it reads.

    Diagnostic, not an error: a weight that legitimately does not track the shift is a valid
    program — which is why the walk is per (family, collection) rather than a membership test.
    """
    session = context_session(ctx)
    state = session.systematics
    factors = state.weight_factors
    if not factors:
        return
    targets = {member.node_id for member in pre_shift.values()}
    for family, nodes in factors:
        if any(targets & cone(session, nid) for nid in nodes):
            # by value, with the factor's own ids: the shipping site filters on them (§2.5's
            # report is about one compiled program, the registry is about the whole Session)
            state.report_shift_after_weight((family, collection), frozenset(nodes))


def _unpack_varied(
    ctx: EventContext,
    name: str,
    collection_name: str,
    varied: Varied,
    points: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """m55: accept a `Varied` collection member only when it fits the registration.

    It is accepted only when it carries EXACTLY the family being registered and its nominal is this
    context's own collection; it unpacks to the `{tag: member}` map the hand form passes, so
    nothing downstream changes. Everything else is refused here, before any label is minted.
    """
    hand = f"collections={{{collection_name!r}: {{tag: record}}}}"
    if points:
        raise GraphedError(
            "points= placements are not accepted beside a Varied collection member "
            f"({collection_name!r}): its members carry no fan-out to prune; pass {hand} for a "
            "placed registration"
        )
    tags = dict(tags_of(varied))
    expected = {"nominal", *(f"{name}_{tag}" for tag in tags.get(name, ()))}
    extra = sorted(set(labels_of(varied)) - expected)
    if set(tags) != {name} or extra:
        raise GraphedError(
            f"collection {collection_name!r}: a Varied member must carry exactly the family "
            f"{name!r} being registered, got families {sorted(tags)} with extra labels {extra}; "
            f"build it as graphed.vary(graphed.nominal(ctx[{collection_name!r}]), {name!r}, ...) "
            f"on the context's central collection with only the tags being added, or pass {hand}"
        )
    current = ctx[collection_name]
    central = member_of(current, "nominal") if isinstance(current, Varied) else current
    nominal = reindex_to(member_of(varied, "nominal"), ctx)
    if nominal.node_id != central.node_id:
        raise GraphedError(
            f"collection {collection_name!r}: the Varied member's nominal is node "
            f"{nominal.node_id}, not this context's {collection_name!r} (node {central.node_id}), "
            "so its members are not shifts of this collection; build it on the context's central "
            f"collection or pass {hand}"
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
            f"{ {key: sorted(value) for key, value in sets.items()} } — one call's collections "
            "must share one tag set"
        )


def two_level(container: Any, label: str) -> Any:
    """§2.1's `factor[L]`: the container's member for L, then.

    When that member is ITSELF a `Varied` (a registered factor computed on shifted objects) — its
    own member for L. The composed ambient weight is therefore always FLAT.

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
    page = session.systematics.universe_page(container)
    path = page.get(label)
    if path is not None:
        member = members_of(container)[path[0]]
        return member if path[1] is None else members_of(member)[path[1]]
    outer_key = key_for(container, label)
    member = members_of(container)[outer_key]
    inner_key = key_for(member, label) if isinstance(member, Varied) else None
    if _settled(container, label):
        page[label] = (outer_key, inner_key)
    return member if inner_key is None else members_of(member)[inner_key]


def _settled(container: Any, label: str) -> bool:
    """Whether a later mint could still move `two_level(container, label)`.

    The one thing that makes the memo above safe to keep across registrations.

    §4.6 reads the point registry at each level, and every read that finds NOTHING is an answer
    the next `vary` can change: an unregistered label falls to the central universe, and an
    unregistered own label is off the container's axes, so a label minted later can start
    resolving onto it. Those are recomputed rather than stored. A label the container carries
    outright never reads the registry at all, and `"nominal"` is registered nowhere by
    construction.
    """
    return _fixed(container, label) and _fixed(member_of(container, label), label)


def _all_settled(session: Any, operands: Sequence[Any], labels: Sequence[str]) -> bool:
    """Whether every resolution the composition just made is one no later mint can move.

    Read off `two_level`'s memo rather than recomputed: that memo stores an answer exactly when
    `_settled` holds, so a page missing a label is that label's unsettled resolution. It must be
    read straight after the composition — a mint can settle a label the composition read before it.
    A composition that is not settled must not be FOLDED onto, because a mint could move an operand
    it already multiplied; it is remade from the original factors instead.
    """
    state = session.systematics
    return all(
        all(label in page for label in labels)
        for page in (
            state.universe_page(operand) for operand in operands if isinstance(operand, Varied)
        )
    )


def _fixed(value: Any, label: str) -> bool:
    if not isinstance(value, Varied):
        return True
    members = members_of(value)
    if label in members:
        return True
    registry = point_registry(value)
    return label in registry and all(own in registry for own in members if own != "nominal")


# ---- §4's composition: linear in the registered factors ------------------------------------
def _member(member: Any, label: str) -> Any:
    """Return the composition's own operand at a label: the member itself."""
    return member


def _times(left: Any, right: Any, label: str) -> Any:
    """Return the composition's multiply, naming the LABEL it failed at.

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
    """Return the ambient at every label.

    The product of each factor's member for that label, read two levels deep so a factor computed
    on shifted objects contributes in the label's own universe.

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
    """Return the balanced product of `factors` at every label.

    `_compose` without the overlays, so a run of product operations between two overlays composes
    exactly as a whole list of them does. An empty run answers with no labels at all, which is how
    a leading or adjacent overlay reads.
    """
    if not factors:
        return {}
    centrals = [two_level(factor, "nominal") for factor in factors]
    central_ids = [member_nodes(central) for central in centrals]
    count = len(centrals)
    tree = _product_tree([project(central, "nominal") for central in centrals], mul)
    complements = _complements(tree, count, mul)
    composed: dict[str, Any] = {}
    for label in labels:
        applied = [two_level(factor, label) for factor in factors]
        varying = [i for i in range(count) if member_nodes(applied[i]) != central_ids[i]]
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
    nominal_ids = [member_nodes(two_level(overlay, "nominal")) for overlay in marks]
    composed: dict[str, Any] = {}
    for label in labels:
        running: Any = runs[0].get(label)
        for index, overlay in enumerate(marks):
            applied = two_level(overlay, label)
            spans = id(overlay) in fixed or (
                member_nodes(applied) != nominal_ids[index]
                and not _placed_elsewhere(overlay, label)
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
    whether a `weight()` read happened to leave a memo — so the composition is remade from the
    list.
    """
    for operand in operands:
        if not isinstance(operand, Varied):
            continue
        registry = point_registry(operand)
        carried = registered_points(operand)
        axes = frozenset(nuisance for own in carried.values() for nuisance, _ in own)
        for label in labels:
            point = registry.get(label)
            if point is None or label in members_of(operand):
                continue
            if restrict(point, axes) and key_for(operand, label) == "nominal":
                return False
    return True


def _placed_elsewhere(overlay: Any, label: str) -> bool:
    """§2.1: whether `label` is a universe ANOTHER family PLACED at this coordinate.

    The point it was placed at carries this overlay's coordinate — the tour's `scale_upup` over a
    `muR` factor and a relative-delta `muF`. The member the placing family declared there is the
    user's value for that point, and the join reads the same member two-level, so the overlay must
    leave it alone for the two outcomes to agree.

    `_route` drops the placing family's OWN axis from an additive placement's point, so a placed
    universe is exactly one that no coordinate of its own point names; a default one-at-a-time
    label and a fanned-out joint both carry theirs (`{name: tag}` and `{name: tag, **foreign}`
    under `f"{name}_{tag}"` and `f"{name}_{tag}__{foreign label}"`). The overlay's own universes
    are its members, placed or not, and it still replaces there.
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


def _product_tree(
    centrals: Sequence[Any], mul: Callable[[Any, Any, str], Any]
) -> dict[tuple[int, int], Any]:
    """`(lo, hi)` -> the product of `centrals[lo:hi]`, split at the midpoint.

    `N-1` multiplies, shared by every label, and a pure function of the registration order.
    """
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
    """Return `out[i]`, the product of every nominal but the `i`-th.

    In O(N) multiplies shared down the tree — so a label exactly one factor varies at costs one
    more. `None` at `count == 1`, where the product of everything else is empty.
    """
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
    """Return the maximal subtrees holding none of `varying`, folded left to right.

    `None` when the varying indices cover every leaf.
    """
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
    """Return the form of a member the composition multiplies.

    `two_level` resolves §2.2's one legal level of nesting, so an ordinary factor arrives here as
    an array. One that varies past that level would compose into an ambient universe that is
    itself a container — which no consumer can read, `graphed.universe(...).node_id` included — so
    it is refused here, inside `vary`, rather than at the read that trips over it.
    """
    if isinstance(member, Varied):
        raise GraphedTypeError(
            "mul",
            capture(),
            f"the ambient weight at variation label {label!r}: this factor's member is itself a "
            f"container over {list(members_of(member))}, one universe deeper than a weight factor "
            "nests",
        )
    return session.form(member)


def _mul_form(session: Any, left: Any, right: Any, label: str) -> Any:
    """Return the product form of two member forms, memoised per Session (`mul_forms`).

    The walk multiplies one factor's handful of member forms at a time, so the distinct pairs stay
    few whatever the union's size; without the memo it would pay a backend inference per multiply.
    The key is the `Form` protocol's own `describe()` and NEVER `str(form)`: a repr may abbreviate
    — awkward's elides the middle of a deep type — so two forms that multiply differently would
    share a key and the memo would admit the very clash the walk exists to refuse.
    """
    key = (left.describe(), right.describe())
    memo = session.systematics.mul_forms
    hit = memo.get(key)
    if hit is not None:
        return hit
    try:
        form = session.backend.op_form("mul", [left, right], {})
    except Exception as exc:  # the clash the composition would raise, named at its own label
        raise GraphedTypeError(
            "mul", capture(), f"the ambient weight at variation label {label!r}: {exc}"
        ) from exc
    memo[key] = form
    return form


def _union(*groups: Sequence[str]) -> tuple[str, ...]:
    out: dict[str, None] = {"nominal": None}
    for group in groups:
        for label in group:
            out.setdefault(label, None)
    return tuple(out)


def _tags_of(name: str, members: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(label[len(name) + 1 :] for label in members if label != "nominal")
