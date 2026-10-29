"""`graphed.vary`'s weight (§2.1) and shift (§2.6a) forms: the calls that write the record."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ...errors import GraphedError
from ..by_label import cone
from ..handles import reindex_to
from ..kinds import Kind
from ..lineage import ambient_of, context_session
from ..registration import check_members, gather_members, record_labels, register
from ..tags import canonical_tag
from ..varied import Varied, labels_of, member_nodes, member_of, members_of, rebuild, tags_of
from .compose import _check_forms, _marks, _resolvable, _union, two_level
from .entries import _expanded, _prefix_slots, live_riders, row_links
from .extension import _extension
from .join import _check_widening, _extend, _overlay
from .materialise import _ambient_operands, _ambient_tags, _carriers, _foldable
from .record import _SLOT, AmbientRecord, Registration, Rider, context_labels, for_vary_link

if TYPE_CHECKING:
    from ...context import EventContext


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


def _tags_of(name: str, members: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(label[len(name) + 1 :] for label in members if label != "nominal")
