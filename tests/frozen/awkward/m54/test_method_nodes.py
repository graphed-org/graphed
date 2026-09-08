"""m54 §2.3 interning and §2.4 result shapes — what a call puts in the store.

Interning is what makes a repeated method call one node rather than N, so the constants have to be
INSIDE the structural identity: equal constants collapse, unequal ones do not, and a list and a
tuple of the same values are the same constant. Determinism extends that across processes-worth of
independent builds: the same program compiled from two sessions is byte-identical.
"""

from __future__ import annotations

import awkward as ak
import pytest
from m54_behavior_fixtures import eager, node_of, op_count, recorded

import graphed
from graphed import GraphedTypeError
from graphed.awkward import gak


def test_the_same_call_twice_is_one_node() -> None:
    session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)

    assert jets.scaled(2.0).node_id == jets.scaled(2.0).node_id
    assert jets.scaled(2.0, offset=1.0).node_id == jets.scaled(2.0, offset=1.0).node_id
    assert lead.near(partner).node_id == lead.near(partner).node_id
    assert node_of(session, jets.scaled(2.0))["name"] == "method"


def test_a_different_constant_or_argument_array_is_a_different_node() -> None:
    _session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
    other = gak.firsts(gak.with_field(jets, jets.pt * 2.0, "pt"), axis=1)

    assert jets.scaled(2.0).node_id != jets.scaled(3.0).node_id
    assert jets.scaled(2.0).node_id != jets.scaled(2.0, offset=1.0).node_id
    assert lead.near(partner).node_id != lead.near(other).node_id
    assert lead.near(partner).node_id != lead.near(partner, threshold=2.0).node_id
    assert jets.scaled(2).node_id != jets.scaled(2.0).node_id  # JSON equality, not Python's


def test_the_keyword_spelling_order_does_not_change_the_node() -> None:
    _session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)

    assert (
        jets.scaled(2.0, offset=1.0, gain=3.0).node_id
        == jets.scaled(2.0, gain=3.0, offset=1.0).node_id
    )
    assert (
        lead.near(other=partner, threshold=1.0).node_id
        == lead.near(threshold=1.0, other=partner).node_id
    )


def test_two_array_keywords_are_ordered_by_name_not_by_spelling() -> None:
    """Sorting the keyword NAMES is not enough: the input list must follow the same order, or the
    two arrays swap silently. The second assertion is the one an order-blind encoder fails."""
    _session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
    third = gak.firsts(gak.with_field(jets, jets.pt * 2.0, "pt"), axis=1)

    assert (
        lead.blend(x=partner, y=third).node_id == lead.blend(y=third, x=partner).node_id
    )
    assert lead.blend(x=partner, y=third).node_id != lead.blend(x=third, y=partner).node_id


def test_a_list_and_a_tuple_of_the_same_constants_are_one_node() -> None:
    _session, jets, _probe = recorded()
    assert jets.combo([2.0, 3.0]).node_id == jets.combo((2.0, 3.0)).node_id
    assert jets.combo([2.0, 3.0]).node_id != jets.combo([2.0, 4.0]).node_id
    assert jets.combo([2.0, 3.0]).node_id != jets.combo([3.0, 2.0]).node_id  # order is semantic


def test_arrays_nested_in_a_dict_constant_follow_the_sorted_key_order() -> None:
    """`sort_keys=True` hides this: with float values both spellings encode identically whatever
    order the inputs were joined in. Only arrays under the keys expose an input list built in
    spelling order rather than sorted order."""
    _session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
    third = gak.firsts(gak.with_field(jets, jets.pt * 2.0, "pt"), axis=1)

    assert (
        lead.mixed({"x": partner, "y": third}).node_id
        == lead.mixed({"y": third, "x": partner}).node_id
    )
    assert (
        lead.mixed({"x": partner, "y": third}).node_id
        != lead.mixed({"x": third, "y": partner}).node_id
    )


def test_two_independent_builds_compile_to_identical_ir() -> None:
    def build() -> tuple[object, object]:
        session, jets, probe = recorded()
        lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
        return session, gak.sum(lead.near(partner, threshold=2.0) * jets.scaled(2.0, offset=1.0))

    first_session, first = build()
    second_session, second = build()
    assert bytes(graphed.compile_ir(first_session, first).ir) == bytes(
        graphed.compile_ir(second_session, second).ir
    )


def test_a_tuple_result_is_a_tuple_of_arrays_one_node_each() -> None:
    session, jets, _probe = recorded()
    parts = jets.split()
    reference = eager("Jet").split()

    assert isinstance(parts, tuple)
    assert len(parts) == len(reference) == 2
    assert parts[0].node_id != parts[1].node_id
    for part, expected in zip(parts, reference, strict=True):
        assert isinstance(part, graphed.Array)
        assert node_of(session, part)["name"] == "method"
        assert ak.array_equal(ak.Array(session.materialize(part)), expected)


def test_a_scalar_returning_method_is_refused_and_records_nothing() -> None:
    session, jets, _probe = recorded()
    before = op_count(session)

    with pytest.raises(GraphedTypeError):
        jets.count_scalar()

    assert op_count(session) == before
