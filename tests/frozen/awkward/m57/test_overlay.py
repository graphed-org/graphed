"""m57 §2.1 "overlay": a nominal that is the nominal of a composition this lineage has READ makes
the family's members that composition in their universes — a replacement, never another factor.

The failing direction is the squared ambient: pre-m57 the handle is appended as a new factor and
every universe is multiplied by the whole composition again.
"""

from __future__ import annotations

from m57_dedupe_fixtures import (
    HF_TABLE,
    LF_TABLE,
    MU,
    PU,
    TRIG,
    m57_ambient_nodes,
    m57_ambient_values,
    m57_at,
    m57_base,
    m57_by_label,
    m57_delta,
    m57_factor,
    m57_node,
    m57_oracle_values,
    m57_overlay,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_table_members,
    m57_trig,
    m57_two_factors,
    m57_weight,
)

import graphed


def test_the_ambient_as_nominal_leaves_every_earlier_universe_untouched() -> None:
    """The overlay's own universe is its member node; every label the ambient already carried keeps
    the node and the value it had before the registration."""
    base = m57_two_factors()
    handle = graphed.weight(base.ctx)
    before_nodes = m57_ambient_nodes(handle)
    before_values = m57_ambient_values(base.session, handle)
    members = m57_scaled(handle, MU)

    registered = m57_delta(base.ctx, "mu", handle)
    weight = graphed.weight(registered)
    after_nodes, after_values = m57_ambient_nodes(weight), m57_ambient_values(base.session, weight)

    assert {label: after_nodes[label] for label in before_nodes} == before_nodes
    assert {label: after_values[label] for label in before_values} == before_values
    for tag, member in members.items():
        label = f"mu_{tag}"
        assert after_nodes[label] == m57_at(member, label).node_id, label


def test_a_join_and_a_new_factor_around_the_overlay_give_the_one_sf_oracle() -> None:
    """§1(e)'s two idioms in sequence: the overlay keeps the names of the factors it was read over,
    a factor registered after it multiplies its result, and a later family joining one of the
    absorbed centrals is still one SF."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    pu_members, trig_members = m57_scaled(pu, PU), m57_scaled(trig, TRIG)
    hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)

    after = m57_weight(m57_weight(ctx, "pu", pu, pu_members), "hf", sf, hf)
    handle = graphed.weight(after)
    after = m57_delta(after, "mu", handle)
    after = m57_weight(after, "trig", trig, trig_members)
    after = m57_weight(after, "lf", sf, lf)
    weight = graphed.weight(after)
    ops = [
        m57_factor("pu", pu, pu_members),
        m57_factor("hf", sf, hf, **m57_by_label("lf", lf)),
        m57_overlay("mu", m57_scaled(handle, MU)),
        m57_factor("trig", trig, trig_members),
    ]

    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_relative_delta_members_over_another_familys_labels_compose() -> None:
    """The members are built FROM the ambient, so their coordinate on the other family is
    composition: no joint is minted, the overlay's universe is exact and the other family's is
    untouched."""
    base = m57_two_factors()
    handle = graphed.weight(base.ctx)
    before = m57_ambient_values(base.session, handle)
    members = m57_scaled(handle, MU)

    registered = m57_delta(base.ctx, "mu", handle)
    weight = graphed.weight(registered)
    values = m57_ambient_values(base.session, weight)

    assert not [label for label in graphed.labels(weight) if "__" in label]
    for tag, member in members.items():
        assert values[f"mu_{tag}"] == m57_ambient_values(base.session, member)["nominal"], tag
    assert values["hf_up"] == before["hf_up"]
    assert m57_node(weight) == m57_node(handle)
