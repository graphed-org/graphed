"""m54 §2.5 — a `Varied` receiver or argument maps the call over the label union.

Per-universe VALUES are the assertion, not just the label set: an implementation that fanned the
labels out but called every member on the nominal operand would produce the right labels and the
wrong arrays, which is exactly the silent-collapse failure the vary arc exists to prevent.
"""

from __future__ import annotations

import awkward as ak
from m54_behavior_fixtures import eager, eager_jets, recorded, varied_jets

import graphed
from graphed.awkward import gak

LABELS = ("nominal", "jes_up", "jes_down")


def test_a_varied_receivers_method_is_a_bound_method() -> None:
    _session, jets, _probe = recorded()
    shifted = varied_jets(jets)
    assert isinstance(shifted.scaled, graphed.BoundMethod)
    assert not isinstance(shifted.scaled, tuple)


def test_a_varied_receiver_fans_the_call_over_its_labels() -> None:
    session, jets, probe = recorded()
    shifted = varied_jets(jets)
    partner = gak.firsts(probe, axis=1)
    e_partner = ak.firsts(eager("Probe"), axis=1)

    result = gak.firsts(shifted, axis=1).deltaR(partner)
    assert set(graphed.labels(result)) == set(LABELS)
    for label in LABELS:
        got = ak.Array(session.materialize(graphed.universe(result, label)))
        expected = ak.firsts(eager_jets(label), axis=1).deltaR(e_partner)
        assert ak.array_equal(got, expected, equal_nan=True), label


def test_a_varied_argument_fans_the_call_over_its_labels() -> None:
    session, jets, probe = recorded()
    shifted = varied_jets(jets)
    lead = gak.firsts(probe, axis=1)
    e_lead = ak.firsts(eager("Probe"), axis=1)

    result = lead.near(gak.firsts(shifted, axis=1), threshold=2.0)
    assert set(graphed.labels(result)) == set(LABELS)
    for label in LABELS:
        got = ak.Array(session.materialize(graphed.universe(result, label)))
        expected = e_lead.near(ak.firsts(eager_jets(label), axis=1), threshold=2.0)
        assert ak.array_equal(got, expected), label


def test_a_constant_only_call_on_a_varied_receiver_varies_too() -> None:
    session, jets, _probe = recorded()
    result = varied_jets(jets).scaled(2.0, offset=1.0)

    assert set(graphed.labels(result)) == set(LABELS)
    nominal = ak.Array(session.materialize(graphed.nominal(result)))
    up = ak.Array(session.materialize(graphed.universe(result, "jes_up")))
    assert ak.array_equal(nominal, eager_jets("nominal").scaled(2.0, offset=1.0))
    assert ak.array_equal(up, eager_jets("jes_up").scaled(2.0, offset=1.0))
    assert not ak.array_equal(nominal, up)  # the shift reached the call, not just the label


def test_a_tuple_result_over_a_varied_receiver_is_a_tuple_of_varied() -> None:
    session, jets, _probe = recorded()
    parts = varied_jets(jets).split()

    assert isinstance(parts, tuple)
    assert len(parts) == 2
    for index, part in enumerate(parts):
        assert set(graphed.labels(part)) == set(LABELS)
        for label in LABELS:
            got = ak.Array(session.materialize(graphed.universe(part, label)))
            assert ak.array_equal(got, eager_jets(label).split()[index]), (label, index)
