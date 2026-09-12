"""m57 §2.7: `graphed.explain(ctx)` — how a user's sources of uncertainty became these variations.

The record is derived from what the context already keeps: the family registrations, the riders, the
link chain from the root and the labels' registered points. It re-decides nothing and mints nothing
beyond what a `weight` read mints, and its text carries no node id, so two Sessions of the same
program render byte-identically.

The layout these legs read is §2.7's and §3's vocabulary: `families` / `operations` / `variations`; a
family's `kind`, `tags`, `links`, `entry`, `placements` and its relation sets (`composes_with`,
`shares_with`, `fans_out_over`, `independent_of`); an operation's `position`, `kind`, `families`,
`links` and `fixed` mark.
"""

from __future__ import annotations

from typing import Any

from m57_dedupe_fixtures import (
    HF_TABLE,
    JES,
    MET_CUT,
    MF_TABLE,
    PU,
    TRIG,
    m57_base,
    m57_delta,
    m57_explain,
    m57_node_count,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_shifted,
    m57_table_members,
    m57_tour,
    m57_trig,
    m57_two_factors,
    m57_weight,
)

import graphed


def _capstone(*, weight_first: bool) -> tuple[Any, Any]:
    """`jes` shifts the jets AND swaps the SF table by name identity; `hf` is a weight family over
    those jets; `pu` is a pure weight off the multiplicity, which the shift cannot move.

    `weight_first=False` registers `hf` BEFORE the shift — §2.5's shift-after-weight order.
    """
    session, ctx = m57_base()
    jets = ctx["Jet"]
    pu = m57_pu(jets)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    if weight_first:
        shifted = m57_shifted(after, "jes", JES)
        sjets = shifted["Jet"]
        central = m57_sf(sjets)
        registered = m57_weight(shifted, "jes", central, m57_table_members(sjets, MF_TABLE))
        return session, m57_weight(registered, "hf", central, m57_table_members(sjets, HF_TABLE))
    flat = m57_sf(jets)
    registered = m57_weight(after, "hf", flat, m57_table_members(jets, HF_TABLE))
    return session, m57_shifted(registered, "jes", JES)


def test_explain_reports_each_familys_entry_form_and_every_labels_origin() -> None:
    """On the §1 chain the joined family names the factor's other family, the overlay names the two
    families it was read over, the placement reports its point, and every label the ambient carries
    has an origin — the overlay's own universes included."""
    tour = m57_tour()
    explanation = m57_explain(tour.ctx)
    families = explanation.families
    labels = set(graphed.labels(graphed.weight(tour.ctx)))

    assert families["lf"].entry.kind == "join"
    assert "hf" in families["lf"].entry.families
    assert families["mu"].entry.kind == "overlay"
    assert set(families["mu"].entry.families) == {"pu", "hf"}
    assert {"hf": "up", "mu": "up"} in [dict(point) for point in families["lf"].placements]
    assert set(explanation.variations) == labels
    assert all(explanation.variations[label] is not None for label in labels)
    assert "mu" in str(explanation.variations["mu_up"])


def test_explain_reports_the_composes_with_sets_of_the_chain() -> None:
    """ "Composes with" is the set of weight families a family shares NO registered point with, so the
    placed diagonal takes `mu` out of `hf`'s set and leaves `pu` there alone."""
    tour = m57_tour()
    families = m57_explain(tour.ctx).families

    assert families["hf"].composes_with == {"pu"}
    assert families["mu"].composes_with == {"pu", "lf"}
    assert families["pu"].composes_with == {"hf", "lf", "mu"}


def test_explain_reports_a_joined_family_as_sharing_the_factor() -> None:
    """Two families on one factor are two values of one operation: they SHARE it, and their joint is
    absent for that reason rather than because it is a product."""
    tour = m57_tour()
    families = m57_explain(tour.ctx).families

    assert families["lf"].shares_with == {"hf"}
    assert families["hf"].shares_with == {"lf"}
    assert "hf" not in families["lf"].composes_with
    assert "lf" not in families["hf"].composes_with


def test_explain_reports_the_links_the_lineage_took() -> None:
    """The registering context's links and each entry's row-space links are the links the lineage
    actually took — read off the record's link fields, not off a rendering."""
    for projected in (False, True):
        base = m57_two_factors()
        child = graphed.universe(base.ctx, "hf_up") if projected else base.ctx[base.met.pt > MET_CUT]
        lf = {tag: graphed.reindex_to(member, child) for tag, member in base.hf_members.items()}
        registered = m57_weight(child, "lf", graphed.reindex_to(base.sf, child), lf)
        explanation = m57_explain(registered)
        expected = ("project", "hf_up") if projected else ("mask", None)

        assert expected in tuple(explanation.families["lf"].links), projected
        assert explanation.families["pu"].links == (), projected
        assert all(expected in tuple(operation.links) for operation in explanation.operations), projected


def test_explain_marks_a_fixed_overlay_and_names_the_universe_that_fixed_it() -> None:
    """Below a projection into an overlay's own universe, and below a further projection, the overlay's
    operation record is marked fixed and its line names the universe that fixed it."""
    _session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    after = m57_delta(after, "mu", graphed.weight(after))
    after = m57_weight(after, "trig", trig, m57_scaled(trig, TRIG))
    outer = graphed.universe(after, "mu_up")
    deeper = m57_weight(
        outer,
        "mf",
        m57_sf(jets, 0.5) * 1.0,
        {tag: graphed.reindex_to(member, outer) for tag, member in m57_table_members(jets, MF_TABLE).items()},
    )
    inner = graphed.universe(deeper, "mf_up")

    for context in (outer, inner):
        explanation = m57_explain(context)
        fixed = [op for op in explanation.operations if op.fixed]
        assert len(fixed) == 1
        assert "mu" in fixed[0].families
        assert "mu_up" in str(fixed[0])


def test_explain_keeps_the_relations_of_a_family_whose_label_the_projection_dropped() -> None:
    """Relations and placements quantify over the family's registered points on the lineage, not over
    the labels this context carries, so a projection that drops a label drops neither."""
    tour = m57_tour()
    at_parent = m57_explain(tour.ctx).families
    projected = graphed.universe(tour.ctx, "pu_up")
    here = m57_explain(projected).families

    assert "mu_up" not in graphed.labels(projected)  # the projection really did drop it
    assert here["hf"].composes_with == at_parent["hf"].composes_with
    assert here["mu"].composes_with == at_parent["mu"].composes_with
    assert [dict(point) for point in here["lf"].placements] == [
        dict(point) for point in at_parent["lf"].placements
    ]


def test_explain_reports_the_capstones_fanout_and_independence() -> None:
    """The capstone's weight family fans out over the shift it reads and composes with the pure-weight
    family; the pure-weight family is independent of the shift and never composes with it."""
    _session, ctx = _capstone(weight_first=True)
    families = m57_explain(ctx).families

    assert families["hf"].fans_out_over == {"jes"}
    assert "pu" in families["hf"].composes_with
    assert families["pu"].independent_of == {"jes"}
    assert "jes" not in families["pu"].composes_with


def test_explain_reports_a_weight_registered_before_the_shift_as_reading_moved_objects() -> None:
    """A weight family registered BEFORE the shift of the objects its central read is reported as
    reading objects a later shift moved — never as independent of that shift."""
    _session, ctx = _capstone(weight_first=False)
    families = m57_explain(ctx).families

    assert "jes" not in families["hf"].independent_of
    assert "jes" in str(families["hf"])
    assert families["pu"].independent_of == {"jes"}  # the live control: this one IS independent


def test_explain_mints_no_more_than_a_weight_read() -> None:
    """`explain` reads the ambient exactly as `weight(ctx)` does and mints nothing else — before and
    after a further registration."""
    tour = m57_tour()
    graphed.weight(tour.ctx)
    after_read = m57_node_count(tour.session)
    m57_explain(tour.ctx)

    assert m57_node_count(tour.session) == after_read
    extra = m57_sf(tour.jets, 0.5) * 1.0
    further = m57_weight(tour.ctx, "nf", extra, m57_scaled(extra, PU))
    graphed.weight(further)
    after_second_read = m57_node_count(tour.session)
    m57_explain(further)
    assert m57_node_count(tour.session) == after_second_read


def test_the_explain_text_is_byte_identical_across_two_sessions() -> None:
    """The text carries no integer node id, so the same program renders identically in two Sessions
    whose ids cannot coincide — the second mints an unrelated node before the program."""
    first = m57_tour()
    second = m57_tour(offset=1)  # an unrelated mint first, so the ids cannot coincide

    first_nodes = graphed.member_of(graphed.weight(first.ctx), "nominal").node_id
    second_nodes = graphed.member_of(graphed.weight(second.ctx), "nominal").node_id

    assert first_nodes != second_nodes  # the ids really do differ
    assert str(m57_explain(first.ctx)) == str(m57_explain(second.ctx))
