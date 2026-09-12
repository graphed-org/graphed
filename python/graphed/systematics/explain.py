"""`graphed.explain` (§2.7): what varies here, how each family entered, and how they relate.

A pure READER of what the context already keeps — the lineage's registrations, the ambient's riders
through `ambient_entries`, the link chain from the root and the Session's point registry. It decides
nothing and records nothing, which is what makes calling it unobservable.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import GraphedError
from .ambient import _live_factors, _live_riders, _Registration, _row_links, ambient_entries
from .kinds import Kind
from .registration import _member_nodes
from .varied import labels_of

if TYPE_CHECKING:
    from ..context import EventContext


@dataclass(frozen=True, slots=True)
class Entry:
    """§2.7(a): how one family entered this context's variations."""

    #: ``"factor"`` (a new one), ``"join"``, ``"overlay"`` or ``"shift"``
    kind: str
    #: the families the form names — the other families on the factor a join entered, the families an
    #: overlay was read over. Empty for a new factor, and for a shift, which names collections
    families: tuple[str, ...] = ()
    #: the collections a SHIFT form varies
    collections: tuple[str, ...] = ()

    def __str__(self) -> str:
        if self.kind == "shift":
            return f"shifts {', '.join(self.collections)}"
        if self.kind == "join":
            named = ", ".join(self.families)
            return f"joins the factor carrying {named}" if named else "joins a factor"
        if self.kind == "overlay":
            return f"an overlay over {', '.join(self.families) or 'the composition it was read over'}"
        return "a new factor"


@dataclass(frozen=True, slots=True)
class Family:
    """§2.7: one ``graphed.vary`` registration, as ``graphed.explain`` reports it."""

    #: the family name and its coordinates
    name: str
    kind: Kind
    tags: tuple[str, ...]
    #: the ROW-SPACE links from the root down to the context that registered it, as ``(kind, label)``
    #: pairs — ``("mask", None)``, ``("project", "hf_up")``; empty at the root
    links: tuple[tuple[str, str | None], ...]
    #: how it entered
    entry: Entry
    #: the POINT of each universe this family PLACED at coordinates of other families
    placements: tuple[tuple[tuple[str, str], ...], ...]
    #: the WEIGHT families whose coordinates appear in none of this one's minted labels — the ones
    #: it COMPOSES with rather than fanning out over, which is why their joint is not a universe
    composes_with: frozenset[str]
    #: the families this one SHARES its operation with — two values of ONE weight (§2.1's join),
    #: never a product, which is why their joint is not a universe either
    shares_with: frozenset[str]
    #: the SHIFT families registered AFTER this weight family read the objects they move (§2.5's
    #: diagnostic): its members carry the pre-shift value, so it is neither fanned out nor independent
    reads_shifted_by: frozenset[str]
    #: the SHIFT families a minted label of this one carries a coordinate of: the shifted objects
    #: this family read, so its members fan out over them
    fans_out_over: frozenset[str]
    #: the SHIFT families no minted label of this one names — it does not read those objects. A
    #: shift is never COMPOSED with: reading it or not reading it are the only two cases.
    independent_of: frozenset[str]

    def __str__(self) -> str:
        placed = "".join(f", placing a universe at {_render_point(point)}" for point in self.placements)
        relations = "".join(
            f"; {phrase} {', '.join(sorted(names))}"
            for phrase, names in (
                ("shares the factor with", self.shares_with),
                ("fans out over", self.fans_out_over),
                ("reads objects later shifted by", self.reads_shifted_by),
                ("independent of", self.independent_of),
                ("composes with", self.composes_with),
            )
            if names
        )
        return (
            f"{self.name} ({self.kind.name}) {list(self.tags)} at {_render_links(self.links)}: "
            f"{self.entry}{placed}{relations}"
        )


@dataclass(frozen=True, slots=True)
class Operation:
    """§2.7(b): one live ambient operation, read from its rider."""

    #: its index in the order the composition applies the operations — the slot is process-wide, so
    #: the position is what a second Session of the same program renders identically
    position: int
    slot: int
    kind: str
    #: the families this operation carries, as ``{name: coordinates}`` — a join's other families too
    families: Mapping[str, tuple[str, ...]]
    #: the row-space links this operation came through, oldest first, as ``(kind, label)`` pairs
    links: tuple[tuple[str, str | None], ...]
    #: the entry's member node ids. The RECORD carries them; the text never prints them, which is
    #: what makes the rendering byte-identical across two Sessions of one program.
    nodes: tuple[int, ...]
    #: §2.3/§2.7(b): whether a projection into this OVERLAY's own universe FIXED it — there every
    #: value is that universe, so the composition replaces with it at every label
    fixed: bool = False
    #: the universe that fixed it, which its line names (`None` for every other operation)
    fixed_at: str | None = None

    def __str__(self) -> str:
        carries = ", ".join(f"{name}{list(tags)}" for name, tags in self.families.items())
        # An entry the registering context still holds has crossed no row space, which is a fact
        # about it, not a missing field — so it is said rather than left as a placeholder.
        through = f"via {_render_links(self.links)}" if self.links else "registered here"
        # §2.7(b): a fixed overlay is MARKED, and the mark names the universe that fixed it — which
        # a later link does not, and the rider is where that universe is recorded
        mark = f", fixed at {self.fixed_at}" if self.fixed else ""
        return f"#{self.position} {self.kind}: {carries} {through}{mark}"


@dataclass(frozen=True, slots=True)
class Variation:
    """§2.7(c): one universe this context carries, and where it came from."""

    label: str
    origin: str
    #: the families whose coordinates the label's registered point names
    families: tuple[str, ...]
    point: tuple[tuple[str, str], ...]

    def __str__(self) -> str:
        return f"{self.label}: {self.origin}"


@dataclass(frozen=True, slots=True)
class Explanation:
    """§2.7: how a user's sources of uncertainty became this context's variations.

    ``str()`` renders one line per item under three headings: the families by name in registration
    order, the ambient's operations in the order the composition applies them, and the universes
    carried here by label with their origins.
    """

    families: Mapping[str, Family]
    operations: tuple[Operation, ...]
    variations: Mapping[str, Variation]

    def __str__(self) -> str:
        lines = [
            f"graphed.explain: {len(self.families)} registrations, "
            f"{len(self.operations)} ambient operations, {len(self.variations)} universes",
            "families (registration order)",
            *(f"  {family}" for family in self.families.values()),
            "ambient operations (in composition order)",
            *(f"  {operation}" for operation in self.operations),
            "universes here",
            *(f"  {variation}" for variation in self.variations.values()),
        ]
        return "\n".join(lines)


def _render_point(point: Sequence[tuple[str, str]]) -> str:
    return "{" + ", ".join(f"{name}: {tag}" for name, tag in point) + "}"


def _render_links(links: Sequence[tuple[str, str | None]]) -> str:
    """The link chain in words — what a reader of the line needs, where the record keeps the pairs."""
    return "/".join(_link_name(link) for link in links) or "the root"


def _link_name(link: tuple[str, str | None]) -> str:
    kind, label = link
    if kind == "mask":
        return "cut"
    return "nominal" if label == "nominal" else f"universe:{label}"


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
    from ..context import EventContext  # noqa: PLC0415  (import cycle: the class it guards)

    if not isinstance(ctx, EventContext):
        raise GraphedError("graphed.explain reads an event context")
    registrations = _lineage_registrations(ctx)
    ambient = ctx._ambient_weight()
    # a row-space change hands its child ONE composed member, which carries no labels of its own:
    # the universe this context is inside is its nominal, and that is the universe it carries
    carried: dict[str, None] = dict.fromkeys(labels_of(ambient) or ("nominal",))
    for collection in ctx._collections.values():
        carried.update(dict.fromkeys(labels_of(collection)))
    points = ctx._session._points
    variations: dict[str, Variation] = {}
    for label in carried:
        point = tuple((name, tag) for name, tag in points.get(label, ()))
        variations[label] = _origin_of(label, point, registrations)
    # §2.7: the relations and the placements quantify over the family's REGISTERED POINTS on the
    # lineage, never over the labels this context happens to carry — a projection drops labels, and
    # what a family does with another is a fact about the registrations, not about where it is read.
    registered = [
        _origin_of(label, tuple((name, tag) for name, tag in point), registrations)
        for label, point in points.items()
        if _minted_by(label, registrations) is not None
    ]
    minted: dict[str, set[str]] = {registration.name: set() for registration in registrations}
    # §2.7: two families SHARE a registered point when one point carries both their coordinates —
    # a symmetric relation over the points as REGISTERED, where a placement's own axis is not one of
    # its coordinates (`_route` spells a placed universe by the point of the OTHERS), which is why
    # the tour's diagonal takes `hf` out of `mu`'s composes-with and leaves `lf` in it
    sharing: set[tuple[str, str]] = set()
    for variation in registered:
        own = _minted_by(variation.label, registrations)
        if own is not None:
            minted.setdefault(own.name, set()).add(variation.label)
        names = [name for name, _tag in variation.point]
        sharing.update((one, other) for one in names for other in names if one != other)
    kinds: dict[str, Kind] = {}
    for registration in registrations:
        kinds.setdefault(registration.name, registration.kind)
    # §2.1: two families on ONE operation are two values of one weight; the riders are where that
    # is recorded, and it is why their joint is absent — not a product, so not "composes with". Read
    # off the LIVE operations, never the listing, which shows an adopted head's product run as the
    # one operation its node is: a run is a product, and its families share nothing.
    shared: dict[str, tuple[str, ...]] = {}
    riders = _live_riders(ctx)
    for slot in _live_factors(ctx)[1]:
        on_slot = riders[slot].families
        for name in on_slot:
            shared[name] = tuple(other for other in on_slot if other != name)
    # §2.5's registry, by (weight family, collection): the shift that moved the objects a weight
    # family had already read, which is neither fanning out nor independence
    flagged: dict[str, tuple[str, ...]] = {}
    for family_name, collection in ctx._session._shift_after_weight:
        flagged[family_name] = tuple(
            dict.fromkeys(
                (
                    *flagged.get(family_name, ()),
                    *(
                        shift.name
                        for shift in registrations
                        if shift.kind == Kind.SHIFT and collection in shift.varies
                    ),
                )
            )
        )
    relations = {
        registration.name: _relations(registration, kinds, minted, registered, shared, flagged, sharing)
        for registration in registrations
    }
    families = {
        registration.name: Family(
            registration.name,
            registration.kind,
            registration.tags,
            _row_links(registration.context),
            Entry(registration.form, registration.names, registration.varies),
            tuple(
                variation.point
                for variation in registered
                if variation.origin.startswith(f"{registration.name} placed")
            ),
            *relations[registration.name],
        )
        for registration in registrations
    }
    operations = tuple(
        Operation(
            position,
            slot,
            rider.kind,
            dict(rider.families),
            rider.links,
            _member_nodes(entry),
            rider.fixed,
            rider.fixed_at,
        )
        for position, (slot, rider, entry) in enumerate(ambient_entries(ctx))
    )
    return Explanation(families, operations, variations)


def _origin_of(
    label: str, point: tuple[tuple[str, str], ...], registrations: Sequence[_Registration]
) -> Variation:
    """§2.7: where one universe came from, read off its registered point and the family whose
    coordinate spells it — never by re-deciding anything."""
    families = tuple(dict.fromkeys(name for name, _tag in point))
    own = _minted_by(label, registrations)
    if own is None:
        return Variation(
            label, f"a point over {', '.join(families) or 'no registered family'}", families, point
        )
    if any(label == f"{own.name}_{tag}" for tag in own.tags):
        others = tuple(name for name in families if name != own.name)
        if others:  # the placing family's own axis is dropped from a placement's point (§2.1)
            return Variation(
                label, f"{own.name} placed at {_render_point(point)}", (own.name, *others), point
            )
        if own.form == "overlay":
            return Variation(label, f"{own.name}'s own universe, a relative-delta family", (own.name,), point)
        return Variation(label, f"{own.name}, one at a time", (own.name,), point)
    read = tuple(name for name in families if name != own.name)
    return Variation(label, f"{own.name} fanned out over {', '.join(read) or '?'}", families, point)


def _minted_by(label: str, registrations: Sequence[_Registration]) -> _Registration | None:
    """The family whose coordinate SPELLS `label` — its own universe first, then a joint of it —
    or `None` for a label no registration on this lineage minted."""
    return next(
        (r for r in registrations for tag in r.tags if label == f"{r.name}_{tag}"),
        next(
            (r for r in registrations for tag in r.tags if label.startswith(f"{r.name}_{tag}__")),
            None,
        ),
    )


def _relations(
    registration: _Registration,
    kinds: Mapping[str, Kind],
    minted: Mapping[str, set[str]],
    variations: Sequence[Variation],
    shared: Mapping[str, tuple[str, ...]],
    flagged: Mapping[str, tuple[str, ...]],
    sharing: Collection[tuple[str, str]],
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """§2.7: what one WEIGHT family does with every other family, read off the points it registered.

    Two weight families whose coordinates never share a label COMPOSE: their joint is a product,
    not a universe, which is why `hf_up__mu_up` is not in the list. Two that SHARE one operation
    (§2.1's join) have no joint either, for the opposite reason — they are two values of one
    weight, and their product is the squaring m57 removes — so they are reported apart. A shift is
    never composed with: a weight family either read the shifted objects, so a minted label carries
    that shift's coordinate and its members FAN OUT over it, or it did not and is INDEPENDENT of
    it — unless it was registered BEFORE the shift moved the objects it read (§2.5's registry),
    where neither holds and the line names the order instead. A shift family's own line reports
    none of them: the relation is a weight family's.
    """
    if registration.kind != Kind.WEIGHT:
        return frozenset(), frozenset(), frozenset(), frozenset(), frozenset()
    reached = {
        other
        for variation in variations
        if variation.label in minted.get(registration.name, ())
        for other in variation.families
    }
    shares = frozenset(shared.get(registration.name, ()))
    fans = frozenset(name for name, kind in kinds.items() if kind == Kind.SHIFT and name in reached)
    preceded = frozenset(flagged.get(registration.name, ())) - fans
    return (
        frozenset(
            name
            for name, kind in kinds.items()
            if kind == Kind.WEIGHT
            and name != registration.name
            and (registration.name, name) not in sharing
            and name not in shares
        ),
        shares,
        preceded,
        fans,
        frozenset(
            name
            for name, kind in kinds.items()
            if kind == Kind.SHIFT and name not in reached and name not in preceded
        ),
    )
