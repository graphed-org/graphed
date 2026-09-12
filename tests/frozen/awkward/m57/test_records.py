"""m57 §2.4/§3: the records a join writes, and the rider each operation carries.

`variations` and `labels(weight(ctx))` must read the same on the join path as on the append path, the
shift-after-weight diagnostic must still see the joined container's member nodes, and
`ambient_entries` must report each entry's provenance — not the bare re-indexed object, whose own tag
map is empty on a slot no join touched.
"""

from __future__ import annotations

from typing import Any

from m57_dedupe_fixtures import (
    HF_TABLE,
    JES,
    LF_TABLE,
    MET_CUT,
    MF_TABLE,
    MU,
    PU,
    m57_ambient_values,
    m57_base,
    m57_delta,
    m57_entries,
    m57_pu,
    m57_scale,
    m57_scaled,
    m57_sf,
    m57_shifted,
    m57_table_members,
    m57_trig,
    m57_two_factors,
    m57_weight,
)

import graphed
from graphed import Kind, compile_ir


def _at(value: Any, ctx: Any) -> Any:
    return graphed.reindex_to(value, ctx)


def _members_at(members: Any, ctx: Any) -> dict[str, Any]:
    return {tag: graphed.reindex_to(member, ctx) for tag, member in members.items()}


def test_variations_reports_the_extending_family_as_a_weight_and_labels_keep_order() -> None:
    """The join path writes `_weight_tags`, `record_labels` and the label union exactly as the append
    path does, and name identity still unions the kinds."""
    _session, ctx = m57_base()
    jets = ctx["Jet"]
    sf = m57_sf(jets)
    after = m57_weight(ctx, "hf", sf, m57_table_members(jets, HF_TABLE))
    after = m57_weight(after, "lf", sf, m57_table_members(jets, LF_TABLE))

    reported = graphed.variations(after)
    assert set(reported["lf"]) == {"up", "down"}
    assert all(kind is Kind.WEIGHT for kind, _value in reported["lf"].values())
    assert graphed.labels(graphed.weight(after)) == (
        "nominal",
        "hf_up",
        "hf_down",
        "lf_up",
        "lf_down",
    )

    # name identity: a `jes` weight joining a factor inside a jes-shifted context is both kinds
    _session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    central = m57_sf(sjets)
    after = m57_weight(shifted, "hf", central, m57_table_members(sjets, HF_TABLE))
    after = m57_weight(after, "jes", central, m57_table_members(sjets, MF_TABLE))
    kinds = graphed.variations(after)["jes"]
    assert all(kind is Kind.WEIGHT | Kind.SHIFT for kind, _value in kinds.values())


def test_the_shift_after_weight_diagnostic_names_a_family_that_joined_a_factor() -> None:
    """`_weight_factors` carries the joined container's member nodes, so a family that joined a factor
    is reported when the objects its members read are shifted afterwards — as a new factor is."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, trig = m57_sf(jets), m57_trig(met)
    after = m57_weight(ctx, "trig", trig, m57_scaled(trig, PU))
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    after = m57_weight(after, "lf", sf, m57_table_members(jets, LF_TABLE))
    shifted = graphed.vary(
        after,
        "jes",
        collections={"Jet": {tag: m57_scale(jets, factor) for tag, factor in JES.items()}},
    )
    weight = graphed.weight(shifted)
    outputs = [graphed.member_of(weight, label) for label in graphed.labels(weight)]

    reported = compile_ir(session, *outputs).shift_after_weight
    assert ("hf", "Jet") in reported
    assert ("lf", "Jet") in reported  # the JOINED family, named as the factor it extended is
    assert ("trig", "Jet") not in reported  # `trig` reads the MET, which this shift leaves alone


def test_ambient_entries_carry_the_families_kind_and_links_of_the_parents() -> None:
    """A join at a row-space change puts the family on the ANCESTOR's slot; the rider reports it, with
    the link the entry came through, where the re-indexed object's own tag map is empty."""
    for projected in (False, True):
        base = m57_two_factors()
        lf = m57_table_members(base.jets, LF_TABLE)
        child = graphed.universe(base.ctx, "hf_up") if projected else base.ctx[base.met.pt > MET_CUT]
        registered = m57_weight(child, "lf", _at(base.sf, child), _members_at(lf, child))

        entries = list(m57_entries(registered))
        riders = [rider for _slot, rider, _entry in entries]
        families = {name for rider in riders for name in rider.families}
        links = [rider.links for rider in riders]

        assert {"pu", "hf", "lf"} <= families, projected
        assert all(rider.kind == "factor" for rider in riders), projected
        expected_link = ("project", "hf_up") if projected else ("mask", None)
        assert all(expected_link[0] in [kind for kind, _label in rider.links] for rider in riders), (
            projected,
            links,
        )


def test_ambient_entries_list_the_order_the_composition_applies() -> None:
    """The listing is the order `_compose` applies the operations — an overlay at its anchored
    position, before a factor the child registered after it — never the order a lineage walk meets
    them."""
    base = m57_two_factors()
    child = base.ctx[base.met.pt > MET_CUT]
    adopted = graphed.weight(child)
    trig = m57_pu(child["Jet"]) * 0.5
    after = m57_delta(child, "mu", adopted)
    after = m57_weight(after, "trig", trig, m57_scaled(trig, PU))

    entries = list(m57_entries(after))
    kinds = [rider.kind for _slot, rider, _entry in entries]
    positions = [index for index, _record in enumerate(entries)]

    assert kinds == ["factor", "overlay", "factor"]
    assert positions == sorted(positions)
    assert "trig" in entries[-1][1].families


def test_ambient_entries_keep_their_slots_kinds_and_order_across_a_value_free_join() -> None:
    """A join that changes no value changes no slot, kind or position either: the joining family
    simply enters its slot's families."""
    base = m57_two_factors()
    projected = graphed.universe(base.ctx, "hf_up")
    before = [(slot, rider.kind, tuple(sorted(rider.families))) for slot, rider, _e in m57_entries(projected)]
    values = m57_ambient_values(base.session, graphed.weight(projected))

    joined = m57_weight(
        projected,
        "pu2",
        _at(base.pu, projected),
        _members_at(m57_scaled(base.pu, MU), projected),
    )
    after = [(slot, rider.kind, tuple(sorted(rider.families))) for slot, rider, _e in m57_entries(joined)]

    assert [slot for slot, _k, _f in after] == [slot for slot, _k, _f in before]
    assert [kind for _s, kind, _f in after] == [kind for _s, kind, _f in before]
    assert any("pu2" in families for _s, _k, families in after)
    assert m57_ambient_values(base.session, graphed.weight(joined))["nominal"] == values["nominal"]
