"""§2.1's join and overlay outcomes: the named container multiplied in exactly once.

A join of an ancestor's factor extends the adopted head; a join that would widen a factor under an
overlay is refused before anything mints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...errors import GraphedError
from ..lineage import ambient_of, links_below
from ..varied import (
    Varied,
    context_of,
    labels_of,
    member_nodes,
    member_of,
    members_of,
    rebuild,
    tags_of,
)
from .compose import _compose, _marks, _union, two_level
from .entries import _expanded, _stood_for, live_factors, live_riders
from .record import Rider, context_labels

if TYPE_CHECKING:
    from ...context import EventContext

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
