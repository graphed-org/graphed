"""`graphed.explain` (§2.7): what varies here, how each family entered, and how they relate.

A pure READER of what the context already keeps — the lineage's registrations, the ambient's riders
through `ambient_entries`, the link chain from the root and the Session's point registry. It decides
nothing and records nothing, which is what makes calling it unobservable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import GraphedError
from .ambient import _Registration, ambient_entries
from .kinds import Kind
from .registration import _member_nodes
from .varied import labels_of

if TYPE_CHECKING:
    from ..context import EventContext, Link


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
    #: the families this one SHARES its operation with — two values of ONE weight (§2.1's join),
    #: never a product, which is why their joint is not a universe either
    shares_with: tuple[str, ...]
    #: the SHIFT families registered AFTER this weight family read the objects they move (§2.5's
    #: diagnostic): its members carry the pre-shift value, so it is neither fanned out nor independent
    reads_shifted_by: tuple[str, ...]
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
                    ("shares the factor with", family.shares_with),
                    ("fans out over", family.fans_out_over),
                    ("reads objects later shifted by", family.reads_shifted_by),
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
    from ..context import EventContext  # noqa: PLC0415  (import cycle: the class it guards)

    if not isinstance(ctx, EventContext):
        raise GraphedError("graphed.explain reads an event context")
    registrations = _lineage_registrations(ctx)
    ambient = ctx._ambient_weight()
    carried: dict[str, None] = dict.fromkeys(labels_of(ambient))
    for collection in ctx._collections.values():
        carried.update(dict.fromkeys(labels_of(collection)))
    points = ctx._session._points
    variations: list[Variation] = []
    for label in carried:
        if label == "nominal":
            continue
        point = tuple((name, tag) for name, tag in points.get(label, ()))
        variations.append(_origin_of(label, point, registrations))
    # §2.7: the relations and the placements quantify over the family's REGISTERED POINTS on the
    # lineage, never over the labels this context happens to carry — a projection drops labels, and
    # what a family does with another is a fact about the registrations, not about where it is read.
    registered = [
        _origin_of(label, tuple((name, tag) for name, tag in point), registrations)
        for label, point in points.items()
        if _minted_by(label, registrations) is not None
    ]
    minted: dict[str, set[str]] = {registration.name: set() for registration in registrations}
    for variation in registered:
        for name in variation.families:
            minted.setdefault(name, set()).add(variation.label)
    kinds: dict[str, Kind] = {}
    for registration in registrations:
        kinds.setdefault(registration.name, registration.kind)
    # §2.1: two families on ONE operation are two values of one weight; the riders are where that
    # is recorded, and it is why their joint is absent — not a product, so not "composes with"
    shared: dict[str, tuple[str, ...]] = {}
    for _slot, rider, _entry in ambient_entries(ctx):
        for name in rider.families:
            shared[name] = tuple(other for other in rider.families if other != name)
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
        registration.name: _relations(registration, kinds, minted, registered, shared, flagged)
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
                for variation in registered
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
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
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
        return (), (), (), (), ()
    reached = {
        other
        for variation in variations
        if variation.label in minted.get(registration.name, ())
        for other in variation.families
    }
    shares = shared.get(registration.name, ())
    fans = tuple(name for name, kind in kinds.items() if kind == Kind.SHIFT and name in reached)
    preceded = tuple(name for name in flagged.get(registration.name, ()) if name not in fans)
    return (
        tuple(
            name
            for name, kind in kinds.items()
            if kind == Kind.WEIGHT
            and name != registration.name
            and name not in reached
            and name not in shares
        ),
        shares,
        preceded,
        fans,
        tuple(
            name
            for name, kind in kinds.items()
            if kind == Kind.SHIFT and name not in reached and name not in preceded
        ),
    )
