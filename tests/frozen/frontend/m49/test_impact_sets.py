"""§3.4's impact API, §9.1's `graphed.member_of` export, and the two-form operand rejection both
m49 verbs share (§3.4, §12.4(1))."""

from __future__ import annotations

import pytest
from m49_vary_fixtures import (
    flat_projection_program,
    reachable,
    shared_node_topology,
    varied_mask_context_program,
)

import graphed
from graphed import Session
from graphed.errors import GraphedError
from graphed.numpy import NumpyBackend

VERBS = ("impact_by_label", "read_columns_by_label")


def _call(verb: str, operand: object) -> object:
    """Both m49 verbs over one operand; the stats verb additionally takes `read_columns`' own
    `source_nid`, which the rejection must reach before it is used."""
    if verb == "impact_by_label":
        return graphed.impact_by_label(operand)
    return graphed.read_columns_by_label(operand, 0)


def test_impact_reports_the_reachability_difference_per_label() -> None:
    session = Session(NumpyBackend())
    topology = shared_node_topology(session)
    impact = graphed.impact_by_label([topology.total])

    assert tuple(impact) == graphed.labels(topology.total)
    assert impact["nominal"] == ()
    for label in ("jes_up", "jes_down"):
        assert impact[label] == tuple(sorted(impact[label]))


def test_a_node_two_labels_share_lands_in_both_impact_sets() -> None:
    session = Session(NumpyBackend())
    topology = shared_node_topology(session)
    impact = graphed.impact_by_label([topology.total])
    shared = topology.shared

    assert shared in impact["jes_up"]
    assert shared in impact["jes_down"]
    assert shared not in reachable(session, graphed.universe(topology.total, "nominal"))
    assert impact["jes_up"] != impact["jes_down"]
    assert set(impact["jes_up"]) & set(impact["jes_down"]) == {shared}


def test_the_mapping_form_answers_the_same_impact_sets() -> None:
    session = Session(NumpyBackend())
    topology = shared_node_topology(session)
    container = topology.total
    mapping = {label: [graphed.universe(container, label)] for label in graphed.labels(container)}

    assert graphed.impact_by_label(mapping) == graphed.impact_by_label([container])


def test_member_of_is_exported_and_resolves_the_label_union() -> None:
    session = Session(NumpyBackend())
    container = shared_node_topology(session).total
    plain = graphed.universe(container, "nominal")

    assert graphed.member_of(container, "jes_up") is graphed.universe(container, "jes_up")
    assert graphed.member_of(container, "btag_up") is graphed.nominal(container)
    assert graphed.member_of(plain, "jes_up") is plain
    with pytest.raises(KeyError):
        graphed.universe(container, "btag_up")


@pytest.mark.parametrize("verb", VERBS)
def test_a_sequence_operand_must_be_all_varied(verb: str) -> None:
    session = Session(NumpyBackend())
    container = shared_node_topology(session).total
    stray = graphed.universe(container, "nominal")

    with pytest.raises(GraphedError) as caught:
        _call(verb, [container, stray])
    assert type(stray).__name__ in str(caught.value)


@pytest.mark.parametrize("verb", VERBS)
def test_a_mapping_operand_must_map_a_label_to_a_sequence_of_array(verb: str) -> None:
    session = Session(NumpyBackend())
    container = shared_node_topology(session).total

    with pytest.raises(GraphedError) as caught:
        _call(verb, {"jes_up": graphed.universe(container, "jes_up")})
    assert "jes_up" in str(caught.value)

    with pytest.raises(GraphedError) as nested:
        _call(verb, {"jes_up": [container]})
    assert "jes_up" in str(nested.value)
    assert type(container).__name__ in str(nested.value)


@pytest.mark.parametrize("verb", VERBS)
def test_a_varied_member_must_not_itself_be_varied(verb: str) -> None:
    """§2.2 admits nested members, but the per-label walk cannot resolve past one level."""
    nested = varied_mask_context_program().varied

    with pytest.raises(GraphedError) as caught:
        _call(verb, [nested])
    assert type(graphed.nominal(nested)).__name__ in str(caught.value)


@pytest.mark.parametrize("verb", VERBS)
def test_an_operand_of_neither_form_is_refused(verb: str) -> None:
    """A bare container is neither a `Sequence[Varied]` nor a mapping — today an unchecked operand
    reaches for `.session` on it and dies with an `AttributeError` instead."""
    container = flat_projection_program().varied

    with pytest.raises(GraphedError):
        _call(verb, container)
