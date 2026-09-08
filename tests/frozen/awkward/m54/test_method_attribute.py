"""m54 §2.1 — a method is a callable, a property is an array.

The classification is the whole point: `getattr` must split the behavior class's functions from its
properties and from the record's own fields. An implementation that answered `BoundMethod` for every
behavior attribute fails the property leg; one that answered it for every non-field name fails the
shadowed-field leg.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from m54_behavior_fixtures import JetArray, collection, eager, recorded

import graphed
from graphed import GraphedTypeError, Session
from graphed.awkward import gak
from graphed.numpy import NumpyBackend, from_record


def test_a_method_is_a_bound_method_and_a_property_is_an_array() -> None:
    _session, jets, _probe = recorded()

    assert isinstance(jets.scaled, graphed.BoundMethod)  # backend-only behavior
    assert isinstance(jets.deltaR, graphed.BoundMethod)  # vector, globally registered
    assert not isinstance(jets.scaled, graphed.Array)

    assert isinstance(jets.heavy, graphed.Array)  # backend-only PROPERTY
    assert isinstance(jets.rho, graphed.Array)  # vector PROPERTY
    assert isinstance(jets.pt, graphed.Array)  # a plain record field


def test_the_property_side_is_the_closed_set() -> None:
    """Only a data descriptor or a `cached_property` is a property; every other resolving descriptor
    is a method. `partialmethod`, `classmethod` and `singledispatchmethod` are NOT callable, so a
    `callable()`-only rule classifies all three as properties and fails here."""
    _session, jets, _probe = recorded()

    for name in ("combine", "spread", "doubled", "widen"):
        assert isinstance(getattr(jets, name), graphed.BoundMethod), name

    assert isinstance(jets.heavy, graphed.Array)  # property
    assert isinstance(jets.bulk, graphed.Array)  # cached_property
    assert isinstance(jets.MUON_MASS, graphed.Array)  # a bare class constant records a field


def test_every_descriptor_kind_of_method_calls_through_the_recorder() -> None:
    session, jets, probe = recorded()
    lead, partner = gak.firsts(jets, axis=1), gak.firsts(probe, axis=1)
    e_jets, e_lead = eager("Jet"), ak.firsts(eager("Jet"), axis=1)
    e_partner = ak.firsts(eager("Probe"), axis=1)

    calls = [
        (jets.combine(lead, partner), JetArray.combine(e_lead, e_partner)),
        (jets.spread(lead, partner), JetArray.spread(e_lead, e_partner)),
        (jets.doubled(), e_jets.doubled()),
        (jets.widen(3.0), e_jets.widen(3.0)),
    ]
    for recorded_result, expected in calls:
        got = ak.Array(session.materialize(recorded_result))
        assert ak.array_equal(got, expected, equal_nan=True)

    assert ak.array_equal(ak.Array(session.materialize(jets.bulk)), e_jets.bulk)


def test_a_bound_method_names_itself_and_its_receiver() -> None:
    _session, jets, _probe = recorded()
    text = repr(jets.scaled)
    assert "scaled" in text
    assert str(jets.node_id) in text


def test_a_bound_method_is_not_an_operand() -> None:
    _session, jets, _probe = recorded()
    method = jets.scaled

    with pytest.raises(AttributeError):
        _ = method.node_id
    with pytest.raises(TypeError):
        method + 1
    with pytest.raises(TypeError):
        method[0]


def test_a_field_named_like_a_method_is_still_a_field() -> None:
    """Fields resolve FIRST — the pre-m54 rule the classification must not disturb."""
    session, _jets, _probe = recorded()
    shadow = collection(session, "Shadow")

    assert isinstance(shadow.scaled, graphed.Array)
    assert ak.array_equal(
        ak.Array(session.materialize(shadow.scaled)), eager("Shadow")["scaled"]
    )


def test_the_numpy_backend_attribute_path_is_unchanged() -> None:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(6.0), w=np.ones(6))

    assert isinstance(record.pt, graphed.Array)
    with pytest.raises(GraphedTypeError):
        _ = record.nosuch
