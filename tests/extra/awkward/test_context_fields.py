"""An event context lists the names assigned on it, as awkward does (graphed#237)."""

from __future__ import annotations

from typing import Any

import awkward as ak

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano
from graphed.systematics import universe

DATA = {"Jet": [[{"pt": 40.0}], []], "MET": [{"pt": 5.0}, {"pt": 6.0}]}


def _assigned() -> Any:
    session = Session(AwkwardBackend())
    events = gnano.events(from_awkward("events", ak.Array(DATA), session=session))
    events["n_jets"] = gak.num(events.Jet, axis=1)
    events["Jet"] = events.Jet[events.Jet.pt > 30]
    return events


def _awkward_order() -> list[str]:
    events = ak.Array(DATA)
    events["n_jets"] = ak.num(events.Jet, axis=1)
    events["Jet"] = events.Jet[events.Jet.pt > 30]
    return list(events.fields)


def test_fields_append_assigned_names_and_keep_a_replaced_ones_place() -> None:
    events = _assigned()
    assert events.fields == _awkward_order() == ["Jet", "MET", "n_jets"]
    assert gak.fields(events) == events.fields
    assert {name: gak.fields(events[name]) for name in events.fields if name != "n_jets"} == {
        "Jet": ["pt"],
        "MET": ["pt"],
    }


def test_dir_lists_record_fields_and_assigned_names() -> None:
    assert {"Jet", "MET", "n_jets"} <= set(dir(_assigned()))


def test_every_context_kind_lists_the_assigned_names() -> None:
    events = _assigned()
    varied = graphed.vary(
        events, "jes", collections={"Jet": {"up": events.Jet, "down": events.Jet}}
    )
    kinds = {
        "vary": varied,
        "derived": events[gak.num(events.Jet, axis=1) > 0],
        "projected": universe(varied, "jes_up"),
        "with_field": gak.with_field(events, events.MET.pt, "met_pt"),
    }
    assert {kind: ctx.fields for kind, ctx in kinds.items()} == {
        "vary": ["Jet", "MET", "n_jets"],
        "derived": ["Jet", "MET", "n_jets"],
        "projected": ["Jet", "MET", "n_jets"],
        "with_field": ["Jet", "MET", "n_jets", "met_pt"],
    }
