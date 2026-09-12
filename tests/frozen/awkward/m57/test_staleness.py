"""m57 §2.1/§2.5: a record that no longer names the composition it named is refused LOUDLY, and a
refusal that fires after the match leaves the operation list exactly as it found it.

A join that ADDS universes to a nominal member changes the composition an overlay was built over;
neither side can be re-derived, so both orders of that program are refused, each naming the order
that works. Pre-m57 there is no record and no refusal: the program silently squares the SF.
"""

from __future__ import annotations

import pytest
from m57_dedupe_fixtures import (
    HF_TABLE,
    JES,
    LF_TABLE,
    MF_TABLE,
    MU,
    PU,
    m57_ambient_nodes,
    m57_ambient_values,
    m57_base,
    m57_delta,
    m57_factor,
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


def _widening_base() -> tuple[object, object, object, object, object]:
    """A jes shift, `pu`, and `hf` over the SF on the UNSHIFTED jets; returns the session, that
    context, the flat central, the jes-varied central (the widening one) and the shifted jets."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    flat_jets = graphed.nominal(sjets)
    flat, pu = m57_sf(flat_jets), m57_pu(flat_jets)
    after = m57_weight(shifted, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", flat, m57_table_members(flat_jets, HF_TABLE))
    return session, after, flat, m57_sf(sjets), sjets


def test_a_union_that_widens_a_nominal_member_an_overlay_covers_is_refused() -> None:
    """The overlay's members ARE the composition the user read; a join that adds universes under it
    would change that composition underneath them, so it is refused, naming the overlay and the
    order that works."""
    session, after, _flat, widening, sjets = _widening_base()
    with_overlay = m57_delta(after, "mu", graphed.weight(after))
    members = m57_table_members(sjets, LF_TABLE)
    before = m57_node_count(session)  # after the members are built: the call is what is measured

    with pytest.raises(GraphedError) as caught:
        m57_weight(with_overlay, "lf", widening, members)
    message = str(caught.value)

    assert "mu" in message and "lf" in message
    assert "weight(" in message
    assert m57_node_count(session) == before


def test_a_handle_read_before_such_a_join_is_refused_with_and_without_an_intervening_read() -> None:
    """The record is keyed on the whole member map, so a join that leaves the nominal NODE alone is
    still seen to have moved the slot: the stale handle is refused in either read spelling."""
    for extra_read in (False, True):
        _session, after, _flat, widening, sjets = _widening_base()
        handle = graphed.weight(after)
        widened = m57_weight(after, "lf", widening, m57_table_members(sjets, LF_TABLE))
        if extra_read:
            graphed.weight(widened)

        with pytest.raises(GraphedError) as caught:
            m57_delta(widened, "mu", handle)
        message = str(caught.value)

        assert "mu" in message and "weight(" in message, extra_read


def test_the_handle_read_after_the_join_gives_the_shifted_joint() -> None:
    """Read after the join, the handle names the widened composition, so the overlay's joint with the
    shift is the ambient at that shift label rescaled — an SF the shift MOVES, so the joint
    discriminates."""
    session, after, _flat, widening, sjets = _widening_base()
    widened = m57_weight(after, "lf", widening, m57_table_members(sjets, LF_TABLE))
    fresh = graphed.weight(widened)
    registered = m57_delta(widened, "mu", fresh)
    weight = graphed.weight(registered)

    assert "mu_up__jes_up" in graphed.labels(weight)
    assert m57_values(session, graphed.member_of(weight, "mu_up__jes_up")) == m57_values(
        session, graphed.member_of(fresh, "jes_up") * MU["up"]
    )


def test_a_refusal_after_the_match_leaves_the_list_and_its_generations_untouched() -> None:
    """A refusal that fires AFTER the match must leave the operation list, its member nodes and the
    registries as before the call: a handle read before it is still fresh, and the next read is node-
    and label-identical to that one."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    first = graphed.vary(sf, "zz", up=sf * 2.0, down=sf * 0.5)
    clashing = graphed.vary(sf, "zz", up=sf * 4.0, down=sf * 0.5)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", first, m57_table_members(jets, HF_TABLE))
    handle = graphed.weight(after)
    before_nodes = m57_ambient_nodes(handle)

    with pytest.raises(GraphedError):
        m57_weight(after, "lf", clashing, m57_table_members(jets, MF_TABLE))

    # no `weight()` read in between: a read would re-record the handle and hide a spliced list
    registered = m57_delta(after, "mu", handle)
    weight = graphed.weight(registered)
    again = graphed.weight(after)
    ops = [
        m57_factor("pu", pu, m57_scaled(pu, PU)),
        m57_factor("hf", first, m57_table_members(jets, HF_TABLE)),
    ]

    assert m57_ambient_nodes(again) == before_nodes
    assert m57_ambient_values(session, weight)["mu_up"] == m57_values(
        session, graphed.member_of(handle, "nominal") * MU["up"]
    )
    assert m57_ambient_values(session, again) == m57_oracle_values(session, ops, graphed.labels(again))
