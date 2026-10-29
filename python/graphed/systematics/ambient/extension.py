"""§2.1/§2.3: what a weight registration's central NAMES.

A composition a read handed out, a registered factor, one of the refusals, or nothing new — decided
from PROVENANCE (riders, recorded reads), never from an entry as it currently stands.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..lineage import ambient_of, context_session, is_ancestor, links_below, parent_of
from ..varied import Varied, context_of, member_nodes, member_of, members_of
from .entries import _entry_state, live_factors, live_riders
from .record import Rider, _Read

if TYPE_CHECKING:
    from ...context import EventContext


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
    node = member_of(central, "nominal")  # ONE level in, not `two_level`, which peels a second
    if isinstance(node, Varied):  # nested past §2.2's one level; `_check_forms` names it properly
        return None
    _live, slots, overlays = live_factors(ctx)
    riders = live_riders(ctx)
    read = _names_read(ctx, central, slots, overlays, riders)
    if read is not None:
        return read
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


def _names_read(
    ctx: EventContext,
    central: Any,
    slots: Sequence[int],
    overlays: frozenset[int],
    riders: Mapping[int, Rider],
) -> tuple[str, Any] | None:
    """`_extension`'s READ arm: the composition a recorded read handed out that `central` names."""
    held = ambient_of(ctx)
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
