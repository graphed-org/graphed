"""m57 §2.6: no silent change, and the extending program is deterministic.

A program whose registrations never re-use a nominal node must mint EXACTLY the composition it minted
before — which the test pins by rebuilding every universe's product from its own operands and showing
that rebuild mints nothing — and the extending program must serialize byte-identically across two
independent Sessions.
"""

from __future__ import annotations

from typing import Any

from m57_dedupe_fixtures import (
    HF_TABLE,
    LF_TABLE,
    MET_CUT,
    PU,
    m57_ambient_nodes,
    m57_base,
    m57_ir,
    m57_node_count,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_table_members,
    m57_tour,
    m57_weight,
)

import graphed


def test_the_extending_program_serializes_byte_identically_across_two_sessions() -> None:
    """Two Sessions building the §1 chain — two factors, a join, an overlay and a placement — produce
    the same bytes for the same universes."""
    first, second = m57_tour(), m57_tour()
    first_weight, second_weight = graphed.weight(first.ctx), graphed.weight(second.ctx)

    assert graphed.labels(first_weight) == graphed.labels(second_weight)
    assert len(graphed.labels(first_weight)) > 1  # there is something to serialize
    assert m57_ir(first.session, first_weight) == m57_ir(second.session, second_weight)


def test_a_program_that_names_nothing_mints_exactly_the_composition_it_minted_before() -> None:
    """A program whose registrations name no live nominal — including one at a masked child whose
    central is not the parent's — keeps every universe node and mints nothing new: the reference
    products are rebuilt here from the program's own operands."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    pu_members, hf = m57_scaled(pu, PU), m57_table_members(jets, HF_TABLE)
    after = m57_weight(ctx, "pu", pu, pu_members)
    after = m57_weight(after, "hf", sf, hf)
    child = after[met.pt > MET_CUT]
    child_sf = m57_sf(child["Jet"], 0.5)  # NOT the parent's central
    lf = m57_table_members(child["Jet"], LF_TABLE)
    registered = m57_weight(child, "lf", child_sf, lf)
    weight = graphed.weight(registered)
    nodes = m57_ambient_nodes(weight)

    def masked(value: Any) -> Any:
        return graphed.reindex_to(value, child)

    before = m57_node_count(session)
    reference = {
        "nominal": masked(pu * sf) * child_sf,
        "pu_up": child_sf * masked(sf * pu_members["up"]),
        "pu_down": child_sf * masked(sf * pu_members["down"]),
        "hf_up": child_sf * masked(pu * hf["up"]),
        "hf_down": child_sf * masked(pu * hf["down"]),
        "lf_up": masked(pu * sf) * lf["up"],
        "lf_down": masked(pu * sf) * lf["down"],
    }

    assert set(nodes) == set(reference)
    assert nodes == {label: value.node_id for label, value in reference.items()}
    assert m57_node_count(session) == before  # rebuilding the reference minted nothing
