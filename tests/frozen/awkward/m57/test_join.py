"""m57 §2.1 "joins a factor": a nominal that IS a live factor's nominal node joins that factor's
container instead of appending a second copy of it.

The failing direction throughout is the squared central: pre-m57 every registration appends its
nominal as a new factor and the composition multiplies it in again.
"""

from __future__ import annotations

import pytest
from m57_dedupe_fixtures import (
    HF_TABLE,
    JES,
    LF_TABLE,
    MF_TABLE,
    PT_CUT,
    PU,
    TAGS,
    m57_ambient_values,
    m57_base,
    m57_by_label,
    m57_factor,
    m57_node,
    m57_node_count,
    m57_oracle_values,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_shifted,
    m57_table_members,
    m57_values,
    m57_weight,
)

import graphed
from graphed import GraphedError
from graphed.awkward import gak


def test_two_families_on_one_central_give_the_one_sf_oracle_and_the_central_node() -> None:
    """Both families' universes are absolute values of ONE weight: each is its own declared member
    and the nominal universe is the central node itself. Fails on the squared nominal."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf = m57_sf(jets)
    hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)

    registered = m57_weight(m57_weight(ctx, "hf", sf, hf), "lf", sf, lf)
    weight = graphed.weight(registered)
    joined = [m57_factor("hf", sf, hf, **m57_by_label("lf", lf))]

    assert set(graphed.labels(weight)) == {"nominal", *m57_by_label("hf", hf), *m57_by_label("lf", lf)}
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, joined, graphed.labels(weight))
    assert m57_node(weight) == sf.node_id


def test_two_centrals_with_different_shift_coordinates_agree_in_both_orders() -> None:
    """The joined nominal member is the UNION of the two centrals' coordinate universes, so the
    ambient reads the weight at each label's own jets whichever family registered first."""

    def program(shift_first: bool) -> tuple[object, object, list[object]]:
        session, ctx = m57_base()
        shifted = m57_shifted(ctx, "jes", JES)
        sjets = shifted["Jet"]
        flat_jets = graphed.nominal(sjets)
        flat = m57_sf(flat_jets)  # the shared nominal node: the SF on the unshifted jets
        varied = m57_sf(sjets)  # the same node at "nominal", jes universes above it
        hf = m57_table_members(flat_jets, HF_TABLE)  # no jes coordinate
        lf = m57_table_members(sjets, LF_TABLE)  # jes-dependent → fans out

        def register_hf(target: object) -> object:
            return m57_weight(target, "hf", flat, hf)

        def register_lf(target: object) -> object:
            return m57_weight(target, "lf", varied, lf)

        steps = (register_hf, register_lf) if shift_first else (register_lf, register_hf)
        target: object = shifted
        for step in steps:
            target = step(target)
        weight = graphed.weight(target)
        joints = {
            f"lf_{tag}__jes_{direction}": graphed.member_of(lf[tag], f"jes_{direction}")
            for tag in TAGS
            for direction in TAGS
        }
        ops = [m57_factor("hf", varied, hf, **m57_by_label("lf", lf), **joints)]
        return session, weight, ops

    session, weight, ops = program(shift_first=True)
    other_session, other_weight, other_ops = program(shift_first=False)

    # not an empty agreement: the union really does carry the jes-dependent family's joints
    assert {label for label in graphed.labels(weight) if "__" in label}
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))
    assert m57_ambient_values(other_session, other_weight) == m57_oracle_values(
        other_session, other_ops, graphed.labels(other_weight)
    )
    assert set(graphed.labels(weight)) == set(graphed.labels(other_weight))


def test_a_coordinate_both_centrals_declare_with_different_nodes_is_refused_before_minting() -> None:
    """The union cannot pick between two nodes for one coordinate, and refusing after composing to
    decide would mint inside the refused call: the node count is unmoved across it."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    first = graphed.vary(sf, "zz", up=sf * 2.0, down=sf * 0.5)
    clashing = graphed.vary(sf, "zz", up=sf * 4.0, down=sf * 0.5)
    agreeing = graphed.vary(sf, "zz", up=sf * 2.0, down=sf * 0.5)

    # two live factors and NO ambient read since the last of them: the memo is cold, so a build
    # that composed to decide would mint here
    after_pu = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    registered = m57_weight(after_pu, "hf", first, m57_table_members(jets, HF_TABLE))
    clashing_members = m57_table_members(jets, MF_TABLE)
    before = m57_node_count(session)
    with pytest.raises(GraphedError) as caught:
        m57_weight(registered, "lf", clashing, clashing_members)
    message = str(caught.value)

    assert m57_node_count(session) == before
    assert "lf" in message and "zz_up" in message
    assert str(graphed.universe(first, "zz_up").node_id) in message
    assert str(graphed.universe(clashing, "zz_up").node_id) in message
    # the live control: the same shape with the coordinate AGREEING is accepted, and composing it
    # does mint — so the unmoved count above is a decision, not a dead instrument
    accepted = m57_weight(registered, "lf", agreeing, clashing_members)
    m57_ambient_values(session, graphed.weight(accepted))
    assert m57_node_count(session) > before


def test_the_capstone_mints_the_same_labels_and_values_in_both_registration_orders() -> None:
    """A nuisance that shifts the jets AND swaps the SF table joins the family registered on the
    same central: one container, so the order of the two registrations cannot matter."""

    def capstone(weight_first: bool) -> tuple[object, object, list[object]]:
        session, ctx = m57_base()
        shifted = m57_shifted(ctx, "jes", JES)
        sjets = shifted["Jet"]
        central = m57_sf(sjets)
        jes_members = m57_table_members(sjets, MF_TABLE)
        hf_members = m57_table_members(sjets, HF_TABLE)

        def register_jes(target: object) -> object:
            return m57_weight(target, "jes", central, jes_members)

        def register_hf(target: object) -> object:
            return m57_weight(target, "hf", central, hf_members)

        steps = (register_jes, register_hf) if weight_first else (register_hf, register_jes)
        target: object = shifted
        for step in steps:
            target = step(target)
        weight = graphed.weight(target)
        joints = {
            f"hf_{tag}__jes_{direction}": graphed.member_of(hf_members[tag], f"jes_{direction}")
            for tag in TAGS
            for direction in TAGS
        }
        ops = [m57_factor("jes", central, jes_members, **m57_by_label("hf", hf_members), **joints)]
        return session, weight, ops

    session, weight, ops = capstone(weight_first=True)
    other_session, other_weight, other_ops = capstone(weight_first=False)

    assert set(graphed.labels(weight)) == set(graphed.labels(other_weight))
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))
    assert m57_ambient_values(other_session, other_weight) == m57_oracle_values(
        other_session, other_ops, graphed.labels(other_weight)
    )


def test_a_joint_of_two_families_on_one_factor_is_the_containers_cross_member() -> None:
    """At a joint of the two families the ambient is the ONE container's cross member for that
    point — the other family's one-at-a-time member is not also multiplied in."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    central = m57_sf(sjets)
    jes_members = m57_table_members(sjets, MF_TABLE)
    hf_members = m57_table_members(sjets, HF_TABLE)

    registered = m57_weight(m57_weight(shifted, "jes", central, jes_members), "hf", central, hf_members)
    weight = graphed.weight(registered)

    for tag in TAGS:
        for direction in TAGS:
            label = f"hf_{tag}__jes_{direction}"
            oracle = m57_sf(graphed.member_of(sjets, f"jes_{direction}"), HF_TABLE[tag])
            assert m57_values(session, graphed.member_of(weight, label)) == m57_values(session, oracle), label


def test_the_ratio_spellings_placed_joint_carries_both_families_variations() -> None:
    """A joint that carries BOTH families' variations is spelled by building the cross member from
    the other family's varied member and PLACING it at that point; joining the factor makes the
    placed universe one value of the one operation, so nothing else multiplies it."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf = m57_sf(jets)
    hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)
    registered = m57_weight(m57_weight(ctx, "hf", sf, hf), "lf", sf, lf)
    cross = hf["up"] * lf["up"] / sf  # the ratio idiom: hf's varied member times lf's ratio
    placed = graphed.vary(
        registered,
        "x",
        sf,
        is_weight=True,
        points=[("upup", cross), {"x": "upup", "hf": "up", "lf": "up"}],
    )
    weight = graphed.weight(placed)

    assert graphed.points(placed)["x_upup"] == {"hf": "up", "lf": "up"}
    assert m57_values(session, graphed.member_of(weight, "x_upup")) == m57_values(session, cross)
    ops = [m57_factor("hf", sf, hf, **m57_by_label("lf", lf), x_upup=cross)]
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_a_family_extending_a_factor_still_fans_out_over_the_shift_it_reads() -> None:
    """Joining a factor does not make the family's own dependence disappear: members reached through
    the shifted jets mint their joints, each read two-level from the one container."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    flat = m57_sf(graphed.nominal(sjets))
    hf = m57_table_members(graphed.nominal(sjets), HF_TABLE)
    lf = m57_table_members(sjets, LF_TABLE)

    registered = m57_weight(m57_weight(shifted, "hf", flat, hf), "lf", flat, lf)
    weight = graphed.weight(registered)
    joints = {
        f"lf_{tag}__jes_{direction}": graphed.member_of(lf[tag], f"jes_{direction}")
        for tag in TAGS
        for direction in TAGS
    }
    ops = [m57_factor("hf", flat, hf, **m57_by_label("lf", lf), **joints)]

    assert set(joints) <= set(graphed.labels(weight))
    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_a_recomputed_central_with_equal_values_is_a_new_factor() -> None:
    """The comparison is node identity, never value: a different expression with the same values
    stays a product. Today's answer, so this leg is the tree's live control."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf = m57_sf(jets)
    equal = gak.prod(gak.where(jets.pt > PT_CUT, 2.0, 1.0), axis=1)
    hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)

    assert m57_values(session, equal) == m57_values(session, sf)  # equal values
    assert equal.node_id != sf.node_id  # different node
    registered = m57_weight(m57_weight(ctx, "hf", sf, hf), "lf", equal, lf)
    weight = graphed.weight(registered)
    ops = [m57_factor("hf", sf, hf), m57_factor("lf", equal, lf)]

    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_name_identity_extends_the_factor_the_central_names() -> None:
    """A `jes` weight whose nominal is `hf`'s central extends that factor, so inside the jes
    universe the SF appears once — and the fan-out's joints still read from the one container."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    central = m57_sf(sjets)
    hf = m57_table_members(sjets, HF_TABLE)
    jes_members = m57_table_members(sjets, MF_TABLE)

    registered = m57_weight(m57_weight(shifted, "hf", central, hf), "jes", central, jes_members)
    weight = graphed.weight(registered)

    for direction in TAGS:
        label = f"jes_{direction}"
        oracle = m57_sf(graphed.member_of(sjets, label), MF_TABLE[direction])
        assert m57_values(session, graphed.member_of(weight, label)) == m57_values(session, oracle), label
    joints = {f"hf_{tag}__jes_{d}" for tag in TAGS for d in TAGS}
    assert joints <= set(graphed.labels(weight))
    for tag in TAGS:
        for direction in TAGS:
            label = f"hf_{tag}__jes_{direction}"
            oracle = m57_sf(graphed.member_of(sjets, f"jes_{direction}"), HF_TABLE[tag])
            assert m57_values(session, graphed.member_of(weight, label)) == m57_values(session, oracle), label
