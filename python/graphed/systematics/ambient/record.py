"""The ambient's records: `AmbientRecord`, each operation's `Rider`, a `Registration`.

And how a record starts — seeded at a root, copied down a `vary` link.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from ..kinds import Kind
from ..lineage import ContextNode, ambient_of, collections_of, context_session, selection
from ..varied import Varied, labels_of, tags_of
from .compose import _union, two_level

if TYPE_CHECKING:
    from ...context import EventContext

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
