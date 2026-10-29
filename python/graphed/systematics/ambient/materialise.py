"""Reading the ambient (§2.1, §5): the composed container, its memo, and the reads it hands out.

A read is recorded so a later registration can tell which composition its central names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ...array import Array
from ..handles import with_context
from ..lineage import (
    ContextNode,
    ambient_of,
    collections_of,
    context_session,
    parent_of,
    selection,
)
from ..registration import AmbientCarrier, stamp_labels
from ..varied import Varied, labels_of, member_nodes, rebuild, tags_of
from .compose import _all_settled, _compose, _marks, _union, two_level
from .entries import _live_slots, project_riders
from .record import _SLOT, Memo

if TYPE_CHECKING:
    from ...context import EventContext


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
