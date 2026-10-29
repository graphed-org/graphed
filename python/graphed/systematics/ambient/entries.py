"""The live operation list (§2.3): what a context's ambient composes, and where each came from.

Walks back through an adopted head, carries each entry's rider through the row-space links below
it, and re-indexes an entry when an expansion needs its value here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...errors import GraphedError
from ..handles import follow, with_context
from ..lineage import ContextNode, ambient_of, context_session, link_of, links_below, parent_of
from ..varied import context_of
from .compose import two_level
from .record import Rider

if TYPE_CHECKING:
    from ...context import EventContext


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
