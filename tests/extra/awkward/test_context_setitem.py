"""Item assignment on an event context, ``events["sel_jets"] = jets`` (graphed#221)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano
from graphed.context import EventContext
from graphed.errors import GraphedError, GraphedTypeError
from graphed.systematics import labels, nominal, universe

DATA = {
    "Jet": [[{"pt": 40.0}, {"pt": 20.0}], [{"pt": 55.0}], []],
    "Muon": [[{"pt": 1.0}, {"pt": 7.0}], [], [{"pt": 9.0}]],
}


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, gnano.events(from_awkward("events", ak.Array(DATA), session=session))


def _jes(events: Any) -> Any:
    jets = events.Jet
    return graphed.vary(
        events,
        "jes",
        collections={
            "Jet": {
                "up": gak.with_field(jets, jets.pt * 1.5, "pt"),
                "down": gak.with_field(jets, jets.pt * 0.5, "pt"),
            }
        },
    )


def _counts(session: Session, value: Any) -> list[int]:
    return list(session.materialize(gak.num(value, axis=1)).tolist())


def test_a_new_name_reads_back_by_item_and_attribute() -> None:
    session, events = _events()
    events["good_mu"] = events.Muon[events.Muon.pt > 5]
    assert _counts(session, events["good_mu"]) == _counts(session, events.good_mu) == [1, 0, 1]


def test_an_existing_collection_is_replaced() -> None:
    session, events = _events()
    events["Muon"] = events.Muon[events.Muon.pt > 5]
    assert _counts(session, events.Muon) == [1, 0, 1]


def test_a_varied_value_reads_back_per_universe_after_a_vary() -> None:
    session, events = _events()
    events = _jes(events)
    events["sel_jets"] = events.Jet[events.Jet.pt > 30]
    assert labels(events.sel_jets) == ("nominal", "jes_up", "jes_down")
    got = {label: _counts(session, universe(events, label).sel_jets) for label in labels(events)}
    assert got == {"nominal": [1, 1, 0], "jes_up": [1, 1, 0], "jes_down": [0, 0, 0]}


def test_a_later_vary_composes_one_at_a_time() -> None:
    session, events = _events()
    events = _jes(events)
    events["sel_jets"] = events.Jet[events.Jet.pt > 30]
    events["Muon"] = events.Muon[events.Muon.pt > 5]
    muons = events.Muon
    events = graphed.vary(
        events,
        "mes",
        collections={"Muon": {"up": gak.with_field(muons, muons.pt * 2, "pt"), "down": muons}},
        composes_as_union=True,
    )
    assert labels(events) == ("nominal", "jes_up", "jes_down", "mes_up", "mes_down")
    assert labels(events.sel_jets) == ("nominal", "jes_up", "jes_down")
    assert _counts(session, universe(events, "mes_up").sel_jets) == [1, 1, 0]
    assert session.materialize(universe(events, "mes_up").Muon.pt).tolist() == [[14.0], [], [18.0]]


def test_a_weight_reading_the_assigned_field_varies_with_its_shift() -> None:
    session, events = _events()
    events = _jes(events)
    events["sel_jets"] = events.Jet[events.Jet.pt > 30]
    sf = gak.prod(events.sel_jets.pt * 0 + 1.1, axis=1)
    events = graphed.vary(
        events, "btag", sf, is_weight=True, up=sf * 2, down=sf, composes_as_union=True
    )
    ambient = graphed.systematics.weight(events)
    got = {label: session.materialize(universe(ambient, label)).tolist() for label in labels(ambient)}
    assert got["jes_up"] == pytest.approx([1.1, 1.1, 1.0])
    assert got["jes_down"] == pytest.approx([1.0, 1.0, 1.0])
    assert got["btag_up"] == pytest.approx([2.2, 2.2, 2.0])


def test_earlier_derivations_keep_what_they_were_derived_with() -> None:
    session, events = _events()
    mask = gak.num(events.Muon, axis=1) > 0
    before, central, jets = events[mask], nominal(events), events.Jet
    events["Muon"] = events.Muon[events.Muon.pt > 5]
    assert _counts(session, before.Muon) == [2, 1]
    assert _counts(session, central.Muon) == [2, 0, 1]
    assert _counts(session, events[mask].Muon) == [1, 1]
    assert _counts(session, nominal(events).Muon) == [1, 0, 1]
    combined = gak.num(jets, axis=1) + gak.num(events.Muon, axis=1)
    assert session.materialize(combined).tolist() == [3, 1, 1]


def test_with_field_returns_a_new_context_and_leaves_the_old_one() -> None:
    session, events = _events()
    child = gak.with_field(events, events.Muon[events.Muon.pt > 5], "Muon")
    assert isinstance(child, EventContext)
    assert _counts(session, child.Muon) == [1, 0, 1]
    assert _counts(session, events.Muon) == [2, 0, 1]
    assert _counts(session, gak.concatenate([child.Muon, events.Muon], axis=1)) == [3, 0, 2]


def test_with_field_keeps_the_context_registrations() -> None:
    session, events = _events()
    events = _jes(events)
    child = gak.with_field(events, events.Jet[events.Jet.pt > 30], "Jet")
    assert labels(child) == labels(events)
    assert _counts(session, nominal(child.Jet)) == [1, 1, 0]


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (lambda e: 1.0, GraphedTypeError, "takes a deferred array or Varied"),
        (
            lambda e: from_awkward("events", ak.Array(DATA), session=Session(AwkwardBackend())).Muon,
            GraphedError,
            "different Session",
        ),
        (
            lambda e: e[gak.num(e.Muon, axis=1) > 0].Muon,
            GraphedError,
            "compute it from .* or a context it was derived from",
        ),
        (
            lambda e: graphed.vary(e.Muon.pt, "loose", up=e.Muon.pt * 2, down=e.Muon.pt),
            GraphedError,
            r"universes \['loose_up', 'loose_down'\].*graphed\.vary\(events",
        ),
    ],
)
def test_assignments_without_a_meaning_are_refused(value: Any, error: Any, match: str) -> None:
    _, events = _events()
    with pytest.raises(error, match=match):
        events["x"] = value(events)
    with pytest.raises(error, match=match):
        gak.with_field(events, value(events), "x")


def test_a_non_name_key_is_refused() -> None:
    _, events = _events()
    with pytest.raises(GraphedTypeError, match="whole collection by name"):
        events["Muon", "pt"] = events.Muon.pt


def test_another_gak_function_names_the_context() -> None:
    _, events = _events()
    with pytest.raises(GraphedTypeError, match="is an event context, not an array"):
        gak.num(events, axis=1)
