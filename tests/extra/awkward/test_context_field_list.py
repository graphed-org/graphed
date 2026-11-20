"""A field list read through an event context reads the collections it holds (graphed#222)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session, Varied
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano
from graphed.errors import GraphedError
from graphed.systematics import labels, universe

DATA = {"Jet": [[{"pt": 40.0}], []], "MET": [{"pt": 5.0}, {"pt": 6.0}]}


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    events = gnano.events(from_awkward("events", ak.Array(DATA), session=session))
    jets = events.Jet
    up = gak.with_field(jets, jets.pt * 2, "pt")
    return session, graphed.vary(events, "jes", collections={"Jet": {"up": up, "down": jets}})


def _at(session: Session, value: Any, label: str) -> Any:
    return ak.to_list(session.materialize(universe(value, label)))


def test_a_shifted_collection_is_read_in_every_universe() -> None:
    session, events = _events()
    got = events[["MET", "Jet"]]
    assert labels(got) == ("nominal", "jes_up", "jes_down")
    assert _at(session, got, "jes_up") == [
        {"MET": {"pt": 5.0}, "Jet": [{"pt": 80.0}]},
        {"MET": {"pt": 6.0}, "Jet": []},
    ]
    assert _at(session, got, "nominal")[0]["Jet"] == [{"pt": 40.0}]


def test_a_list_of_tree_fields_alone_stays_unvaried() -> None:
    session, events = _events()
    got = events[["MET"]]
    assert not isinstance(got, Varied)
    assert ak.to_list(session.materialize(got)) == [{"MET": {"pt": 5.0}}, {"MET": {"pt": 6.0}}]


def test_an_assigned_name_is_read_in_list_order() -> None:
    session, events = _events()
    events["sel"] = events.Jet[events.Jet.pt > 50]
    got = events[["sel", "MET"]]
    assert _at(session, got, "jes_up") == [
        {"sel": [{"pt": 80.0}], "MET": {"pt": 5.0}},
        {"sel": [], "MET": {"pt": 6.0}},
    ]
    assert _at(session, got, "nominal")[0] == {"sel": [], "MET": {"pt": 5.0}}


def test_derived_and_projected_contexts_read_their_own_collections() -> None:
    session, events = _events()
    selected = events[gak.num(events.Jet, axis=1) > 0]
    assert _at(session, selected[["Jet"]], "jes_up") == [{"Jet": [{"pt": 80.0}]}]
    projected = universe(events, "jes_up")[["Jet"]]
    assert ak.to_list(session.materialize(projected)) == [{"Jet": [{"pt": 80.0}]}, {"Jet": []}]


def test_a_backend_without_the_seam_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _, events = _events()
    monkeypatch.delattr(AwkwardBackend, "with_field")
    with pytest.raises(GraphedError, match=r"holds \['Jet'\] as collections.*events\.<name>"):
        events[["Jet"]]
