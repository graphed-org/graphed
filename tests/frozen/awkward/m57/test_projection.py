"""m57 §2.3: what a central may name at `graphed.universe(ctx, L)`.

A projection keeps the entries' identity, so a central built at the parent still names its factor
there — except the factor that OWNS `L`, whose member at `L` is the universe projected into: naming
it would erase the variation the projection selected, and that is refused. Pre-m57 every one of
these registrations multiplies the factor in again without a word.
"""

from __future__ import annotations

from typing import Any

import pytest
from m57_dedupe_fixtures import (
    HF_TABLE,
    JES,
    LF_TABLE,
    MET_CUT,
    MF_TABLE,
    MU,
    NF_TABLE,
    NU,
    PU,
    TRIG,
    m57_ambient_values,
    m57_base,
    m57_delta,
    m57_factor,
    m57_node,
    m57_oracle_values,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_shifted,
    m57_table_members,
    m57_trig,
    m57_two_factors,
    m57_values,
    m57_weight,
)

import graphed
from graphed import GraphedError


def _at(value: Any, ctx: Any) -> Any:
    return graphed.reindex_to(value, ctx)


def _members_at(members: Any, ctx: Any) -> dict[str, Any]:
    return {tag: graphed.reindex_to(member, ctx) for tag, member in members.items()}


def test_naming_a_factor_that_does_not_own_the_label_joins_it_and_keeps_the_universe() -> None:
    """`pu` named inside `hf_up` joins the `pu` factor: the projected nominal is still the universe
    the projection selected, and the new family's universe multiplies it once."""
    base = m57_two_factors()
    projected = graphed.universe(base.ctx, "hf_up")
    adopted = graphed.weight(projected)
    pu2 = m57_scaled(base.pu, NU)

    registered = m57_weight(projected, "pu2", _at(base.pu, projected), _members_at(pu2, projected))
    weight = graphed.weight(registered)
    ops = [
        m57_factor("pu2", _at(base.pu, projected), _members_at(pu2, projected)),
        m57_factor("hf", _at(base.hf_members["up"], projected), {}),
    ]

    assert m57_node(weight) == adopted.node_id
    assert m57_ambient_values(base.session, weight) == m57_oracle_values(
        base.session, ops, graphed.labels(weight)
    )


def test_naming_the_factor_that_owns_the_label_is_refused() -> None:
    """The owner's member at `L` IS the universe projected into; naming it there would replace the
    selected variation, so it is refused — before and after an unrelated join at the projection, and
    after an expansion at a masked child."""
    for expanded in (False, True):
        base = m57_two_factors()
        source = base.ctx
        if expanded:
            child = base.ctx[base.met.pt > MET_CUT]
            source = m57_weight(
                child,
                "lf",
                _at(base.sf, child),
                _members_at(m57_table_members(base.jets, LF_TABLE), child),
            )
        projected = graphed.universe(source, "hf_up")
        owner = _at(base.sf, projected)
        members = _members_at(m57_table_members(base.jets, MF_TABLE), projected)
        for joined_first in (False, True):
            target = projected
            if joined_first:
                target = m57_weight(
                    projected,
                    "pu2",
                    _at(base.pu, projected),
                    _members_at(m57_scaled(base.pu, NU), projected),
                )
            before = m57_node(graphed.weight(target))
            with pytest.raises(GraphedError) as caught:
                m57_weight(target, "mf", owner, members)
            message = str(caught.value)
            assert "hf_up" in message, (expanded, joined_first)
            assert "weight(" in message, (expanded, joined_first)
            assert m57_node(graphed.weight(target)) == before, (expanded, joined_first)


def test_at_a_placed_label_both_owners_are_refused_and_a_third_factor_joins() -> None:
    """A placed label's point carries two families' coordinates, so it has two owners; ownership is
    decided from the riders' families against the point, never from the label's name."""
    _session, ctx = m57_base()
    jets = ctx["Jet"]
    pu = m57_pu(jets)
    first, second = m57_sf(jets), m57_sf(jets, 1.5) * 1.0  # two distinct SF centrals
    hf = m57_table_members(jets, HF_TABLE)
    lf = m57_table_members(jets, LF_TABLE)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", first, hf)
    after = m57_weight(after, "lf", second, lf)
    cross = hf["up"] * lf["up"]
    after = graphed.vary(
        after,
        "x",
        m57_sf(jets, 0.5),
        is_weight=True,
        points=[("upup", cross), {"x": "upup", "hf": "up", "lf": "up"}],
    )
    projected = graphed.universe(after, "x_upup")
    before = m57_node(graphed.weight(projected))

    for owner in (first, second):
        with pytest.raises(GraphedError):
            m57_weight(
                projected,
                "probe",
                _at(owner, projected),
                _members_at(m57_table_members(jets, MF_TABLE), projected),
            )
    assert m57_node(graphed.weight(projected)) == before
    joined = m57_weight(projected, "pu2", _at(pu, projected), _members_at(m57_scaled(pu, NU), projected))
    assert m57_node(graphed.weight(joined)) == before


def test_a_central_equal_to_the_owners_member_is_refused_while_a_re_derivation_below_a_mask_is_not() -> None:
    """The owner's RECORDED member at `L` is refused however the central was built — at the parent as
    the `up=` node, or at the projection — in either order with an unrelated family and across a mask;
    a member RE-DERIVED below the mask is a different node and stays a new factor."""
    base = m57_two_factors()
    owner_member = base.hf_members["up"]
    probe = m57_table_members(base.jets, MF_TABLE)
    spectator = m57_scaled(base.pu, NU)

    for masked in (False, True):
        source = base.ctx[base.met.pt > MET_CUT] if masked else base.ctx
        projected = graphed.universe(source, "hf_up")
        for spectator_first in (False, True):
            target = projected
            if spectator_first:
                target = m57_weight(
                    projected, "zz", _at(base.pu, projected), _members_at(spectator, projected)
                )
            before = m57_node(graphed.weight(target))
            with pytest.raises(GraphedError) as caught:
                m57_weight(target, "mf", _at(owner_member, target), _members_at(probe, target))
            assert "hf_up" in str(caught.value), (masked, spectator_first)
            assert m57_node(graphed.weight(target)) == before, (masked, spectator_first)

    # the control: below the mask the same expression is a different node, so it stays a factor
    child = base.ctx[base.met.pt > MET_CUT]
    projected = graphed.universe(child, "hf_up")
    re_derived = m57_sf(projected["Jet"], HF_TABLE["up"])
    assert re_derived.node_id != _at(owner_member, projected).node_id
    registered = m57_weight(projected, "mf", re_derived, _members_at(probe, projected))
    ops = [
        m57_factor("hf", _at(owner_member, projected), {}),
        m57_factor("pu", _at(base.pu, projected), {}),
        m57_factor("mf", re_derived, _members_at(probe, projected)),
    ]
    assert m57_ambient_values(base.session, graphed.weight(registered)) == m57_oracle_values(
        base.session, ops, graphed.labels(graphed.weight(registered))
    )


def test_below_a_second_projection_the_owner_of_each_universe_is_refused() -> None:
    """The context is inside every universe it projected into, so ownership is tested against all of
    them: the outer owner named by its recorded member is refused as by its nominal, and so is the
    inner one."""
    base = m57_two_factors()
    outer = graphed.universe(base.ctx, "hf_up")
    inner_central = m57_sf(base.jets, 0.5) * 1.0
    inner_members = m57_table_members(base.jets, MF_TABLE)
    with_inner = m57_weight(outer, "mf", _at(inner_central, outer), _members_at(inner_members, outer))
    inner = graphed.universe(with_inner, "mf_up")
    before = m57_node(graphed.weight(inner))
    probe = m57_table_members(base.jets, NF_TABLE)

    for owner in (base.hf_members["up"], inner_members["up"], inner_central):
        with pytest.raises(GraphedError):
            m57_weight(inner, "probe", _at(owner, inner), _members_at(probe, inner))
    assert m57_node(graphed.weight(inner)) == before
    # the live control: a factor neither universe owns still joins, so the refusals above are
    # decisions rather than a blanket refusal of every central at a stacked projection
    joined = m57_weight(inner, "pu2", _at(base.pu, inner), _members_at(m57_scaled(base.pu, NU), inner))
    assert m57_node(graphed.weight(joined)) == before


def test_a_node_that_is_both_a_nominal_and_the_owners_member_joins_that_factor() -> None:
    """A node that is a live factor's nominal in the central's own row space AND another entry's
    recorded member is decided by the nominal first — a join, in either registration order."""
    for hf_first in (True, False):
        _session, ctx = m57_base()
        jets = ctx["Jet"]
        sf = m57_sf(jets)
        hf = m57_table_members(jets, HF_TABLE)
        shared = hf["up"]  # hf's member at hf_up AND the `g` family's own central

        def register_hf(target: Any, sf: Any = sf, hf: Any = hf) -> Any:
            return m57_weight(target, "hf", sf, hf)

        def register_g(target: Any, shared: Any = shared, jets: Any = jets) -> Any:
            return m57_weight(target, "g", shared, m57_table_members(jets, LF_TABLE))

        target: Any = ctx
        for step in (register_hf, register_g) if hf_first else (register_g, register_hf):
            target = step(target)
        projected = graphed.universe(target, "hf_up")
        before = m57_node(graphed.weight(projected))

        registered = m57_weight(
            projected,
            "probe",
            _at(shared, projected),
            _members_at(m57_table_members(jets, MF_TABLE), projected),
        )
        assert m57_node(graphed.weight(registered)) == before, hf_first


def test_at_a_shift_label_the_re_derived_sf_joins_the_shift_dependent_factor() -> None:
    """Inside `jes_up` the SF re-derived over this context's shifted jets IS the jes-dependent
    factor's recorded member there, and that factor does not own the label: it joins."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    pu = m57_pu(graphed.nominal(sjets))
    central = m57_sf(sjets)
    after = m57_weight(shifted, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", central, m57_table_members(sjets, HF_TABLE))
    projected = graphed.universe(after, "jes_up")
    adopted = graphed.weight(projected)
    recorded = m57_sf(projected["Jet"])
    assert recorded.node_id == graphed.member_of(central, "jes_up").node_id

    probe = m57_table_members(projected["Jet"], MF_TABLE)
    registered = m57_weight(projected, "mf", recorded, probe)
    weight = graphed.weight(registered)
    ops = [m57_factor("pu", _at(pu, projected), {}), m57_factor("mf", recorded, probe)]

    assert m57_node(weight) == adopted.node_id
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_at_a_shift_label_a_join_without_that_coordinate_keeps_the_other_factors_dependence() -> None:
    """Joining a factor that has no coordinate on the shift must not flatten the OTHER factor: the
    projected nominal still reads the SF at the shifted jets, beside the jes-dependent join as the
    positive control."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    pu = m57_pu(graphed.nominal(sjets))
    central = m57_sf(sjets)
    after = m57_weight(shifted, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", central, m57_table_members(sjets, HF_TABLE))
    projected = graphed.universe(after, "jes_up")
    adopted = graphed.weight(projected)
    shifted_sf = m57_sf(projected["Jet"])

    joined = m57_weight(projected, "pu2", _at(pu, projected), _members_at(m57_scaled(pu, NU), projected))
    weight = graphed.weight(joined)
    ops = [
        m57_factor("pu2", _at(pu, projected), _members_at(m57_scaled(pu, NU), projected)),
        m57_factor("hf", shifted_sf, {}),
    ]

    assert m57_node(weight) == adopted.node_id
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))
    # positive control on the same list: the jes-dependent factor joins too, keeping the universe
    control = m57_weight(projected, "mf", shifted_sf, m57_table_members(projected["Jet"], MF_TABLE))
    assert m57_node(graphed.weight(control)) == adopted.node_id


def test_inside_an_overlays_universe_a_post_overlay_join_multiplies_and_a_covered_one_is_refused() -> None:
    """Inside the overlay's own universe every value is that universe, so an operation that would
    land INSIDE the factors it covers is refused; one registered after it multiplies its result, and
    a handle over the whole head anchors behind it."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    prefix = graphed.weight(after)
    covered = graphed.weight(m57_weight(ctx, "pu", pu, m57_scaled(pu, PU)))  # a STRICT prefix read
    after = m57_delta(after, "mu", prefix)
    after = m57_weight(after, "trig", trig, m57_scaled(trig, TRIG))
    projected = graphed.universe(after, "mu_up")
    adopted = graphed.weight(projected)
    probe = m57_table_members(jets, MF_TABLE)

    # a family joining the factor registered AFTER the overlay multiplies the overlay's result
    joined = m57_weight(
        projected, "trig2", _at(trig, projected), _members_at(m57_scaled(trig, NU), projected)
    )
    assert m57_node(graphed.weight(joined)) == adopted.node_id

    # a family joining a factor the overlay COVERS, and a handle read over a strict prefix of them
    for central in (_at(sf, projected), _at(pu, projected), _at(covered, projected)):
        with pytest.raises(GraphedError) as caught:
            m57_weight(projected, "probe", central, _members_at(probe, projected))
        assert "mu_up" in str(caught.value)
    assert m57_node(graphed.weight(projected)) == adopted.node_id

    # the whole-head handle anchors behind the head: the nominal is unmoved and the new universe
    # rescales the projected value
    anchored = m57_delta(projected, "mu2", adopted, NU)
    weight = graphed.weight(anchored)
    assert m57_node(weight) == adopted.node_id
    assert m57_values(session, graphed.member_of(weight, "mu2_up")) == m57_values(session, adopted * NU["up"])
    # a handle read AT the projection is the same overlay
    read_here = m57_delta(projected, "mu3", graphed.weight(projected), NU)
    assert m57_node(graphed.weight(read_here)) == adopted.node_id


def test_below_a_further_projection_the_post_overlay_join_keeps_the_overlays_factor() -> None:
    """The overlay stays fixed through every further row-space link: the post-overlay join still
    keeps its factor, the covered join is still refused, and the refusal names the universe that
    FIXED the overlay, not the last link taken."""
    _session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    prefix = graphed.weight(after)
    after = m57_delta(after, "mu", prefix)
    after = m57_weight(after, "trig", trig, m57_scaled(trig, TRIG))
    outer = graphed.universe(after, "mu_up")
    deeper = m57_weight(
        outer, "mf", m57_sf(jets, 0.5) * 1.0, _members_at(m57_table_members(jets, MF_TABLE), outer)
    )
    inner = graphed.universe(deeper, "mf_up")
    adopted = graphed.weight(inner)

    joined = m57_weight(inner, "trig2", _at(trig, inner), _members_at(m57_scaled(trig, NU), inner))
    assert m57_node(graphed.weight(joined)) == adopted.node_id

    with pytest.raises(GraphedError) as caught:
        m57_weight(inner, "probe", _at(sf, inner), _members_at(m57_table_members(jets, NF_TABLE), inner))
    assert "mu_up" in str(caught.value)
    assert m57_node(graphed.weight(inner)) == adopted.node_id


def test_a_fixed_overlay_stays_fixed_through_a_further_row_space_link() -> None:
    """An overlay a projection fixed is never re-tested against its re-indexed member: after an
    expansion at its own universe and a further link carrying a registration, its universe is still
    its member over the projected value."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    prefix = graphed.weight(after)
    after = m57_delta(after, "mu", prefix)
    after = m57_weight(after, "trig", trig, m57_scaled(trig, TRIG))
    projected = graphed.universe(after, "mu_up")
    adopted = graphed.weight(projected)
    expanded = m57_weight(
        projected, "trig2", _at(trig, projected), _members_at(m57_scaled(trig, NU), projected)
    )
    child = expanded[_at(met, expanded).pt > MET_CUT]
    child_sf = m57_sf(child["Jet"], 0.5) * 1.0
    probe = m57_table_members(child["Jet"], MF_TABLE)
    registered = m57_weight(child, "mf", child_sf, probe)
    weight = graphed.weight(registered)
    head = _at(adopted, child)  # the projected value, masked: it carries the overlay's rescaling

    assert m57_values(session, graphed.member_of(weight, "nominal")) == m57_values(session, head * child_sf)
    assert m57_values(session, graphed.member_of(weight, "mf_up")) == m57_values(session, head * probe["up"])


def test_a_parent_built_projected_nominal_is_an_overlay_after_the_head() -> None:
    """`universe(weight(ctx), L)` built at the parent IS the node the projection adopted, so a family
    whose nominal is that expression is an overlay behind the head — with or without a read in
    between; built above a mask that lies between, it is a member of another ambient and stays a
    factor."""
    base = m57_two_factors()
    parent_member = graphed.universe(graphed.weight(base.ctx), "hf_up")

    for extra_read in (False, True):
        projected = graphed.universe(base.ctx, "hf_up")
        if extra_read:
            graphed.weight(projected)
        adopted = graphed.weight(projected)
        registered = m57_delta(projected, "mu", _at(parent_member, projected))
        weight = graphed.weight(registered)
        assert m57_node(weight) == adopted.node_id, extra_read
        assert m57_values(base.session, graphed.member_of(weight, "mu_up")) == m57_values(
            base.session, adopted * MU["up"]
        ), extra_read

    # the control: the same expression built ABOVE a mask that lies between is a new factor
    child = base.ctx[base.met.pt > MET_CUT]
    extra = m57_pu(child["Jet"]) * 0.5  # a factor the child registers, so its ambient is NOT the parent's
    child = m57_weight(child, "nf", extra, m57_scaled(extra, NF_TABLE))
    below = graphed.universe(child, "hf_up")
    crossed = _at(parent_member, below)
    ops = [
        m57_factor("head", graphed.weight(below), {}),
        m57_factor("mu", crossed, m57_scaled(crossed, MU)),
    ]
    registered = m57_delta(below, "mu", crossed)
    assert m57_ambient_values(base.session, graphed.weight(registered)) == m57_oracle_values(
        base.session, ops, graphed.labels(graphed.weight(registered))
    )


def test_at_the_nominal_projection_a_re_derived_central_joins_while_a_weight_label_stays_a_factor() -> None:
    """A projection to the NOMINAL universe keeps the entries' identity, so a central re-derived from
    the projected collections interns to the parent's node and joins; at a WEIGHT label the same
    registration is a new factor and the variation projected into survives."""
    base = m57_two_factors()
    flat = graphed.nominal(base.ctx)
    re_derived = m57_sf(flat["Jet"])
    probe = m57_table_members(flat["Jet"], LF_TABLE)
    assert re_derived.node_id == base.sf.node_id  # the projection interns to the parent's node

    joined = m57_weight(flat, "lf", re_derived, probe)
    ops = [
        m57_factor("pu", _at(base.pu, flat), {}),
        m57_factor("lf", re_derived, probe),
    ]
    assert m57_ambient_values(base.session, graphed.weight(joined)) == m57_oracle_values(
        base.session, ops, graphed.labels(graphed.weight(joined))
    )

    projected = graphed.universe(base.ctx, "hf_up")
    adopted = graphed.weight(projected)
    weight_probe = m57_table_members(projected["Jet"], LF_TABLE)
    factored = m57_weight(projected, "lf", m57_sf(projected["Jet"], 0.5) * 1.0, weight_probe)
    factor_ops = [
        m57_factor("head", adopted, {}),
        m57_factor("lf", m57_sf(projected["Jet"], 0.5) * 1.0, weight_probe),
    ]
    assert m57_ambient_values(base.session, graphed.weight(factored)) == m57_oracle_values(
        base.session, factor_ops, graphed.labels(graphed.weight(factored))
    )
