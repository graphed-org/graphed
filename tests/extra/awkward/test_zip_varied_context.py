"""A call fanned out over Varied operands inside a dict or list keeps their context (#264)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, gnano
from graphed.systematics import universe
from graphed.systematics.varied import context_of

EAGER = ak.Array(
    {
        "run": [1, 2, 3],
        "Jet": [
            [{"pt": 30.0, "eta": 0.1, "btag": 0.9}],
            [],
            [{"pt": 10.0, "eta": -1.0, "btag": 0.2}, {"pt": 40.0, "eta": 2.0, "btag": 0.5}],
        ],
    }
)


def _varied_events() -> tuple[Session, Any, Any]:
    session = Session(AwkwardBackend())
    ctx = gnano.events(from_awkward("ev", EAGER, session=session), is_data=False)
    jets = ctx.Jet
    members = {t: gak.with_field(jets, jets.pt * f, "pt") for t, f in (("up", 1.5), ("down", 0.5))}
    return session, ctx, graphed.vary(ctx, "jes", collections={"Jet": members})


def test_an_unvaried_zip_keeps_its_context() -> None:
    _session, ctx, _events = _varied_events()
    assert context_of(ak.zip({"pt": ctx.Jet.pt, "eta": ctx.Jet.eta})) is not None


ZIPS = {
    "two varied fields": lambda j: ak.zip({"pt": j.pt, "eta": j.eta}),
    "varied and unvaried": lambda j: ak.zip({"pt": j.pt, "eta": j.eta, "btag": j.btag}),
    "with zeros_like": lambda j: ak.zip({"pt": j.pt, "c": ak.zeros_like(j.pt)}),
    "a list operand": lambda j: ak.concatenate([j.pt, j.eta], axis=1),
}


@pytest.mark.parametrize("build", ZIPS.values(), ids=ZIPS.keys())
def test_a_fanned_out_call_over_nested_varied_operands_keeps_the_context(build: Any) -> None:
    _session, _ctx, events = _varied_events()
    out = build(events.Jet)
    assert isinstance(out, graphed.Varied)
    assert context_of(out) is events


def test_the_zipped_record_assigns_into_the_events_per_universe() -> None:
    session, _ctx, events = _varied_events()
    j = events.Jet
    events["sel"] = ak.zip({"pt": j.pt, "eta": j.eta})
    for label, scale in (("nominal", 1.0), ("jes_up", 1.5), ("jes_down", 0.5)):
        got = ak.Array(session.materialize(universe(events.sel.pt, label)))
        assert got.tolist() == (EAGER.Jet.pt * scale).tolist()


def test_a_zip_of_fields_read_through_no_context_is_still_refused() -> None:
    session, _ctx, events = _varied_events()
    raw = from_awkward("raw", EAGER, session=session)
    loose = graphed.vary(raw.Jet.pt, "s", points={"up": raw.Jet.pt * 2})
    with pytest.raises(graphed.GraphedError, match="not read through an event context"):
        events["sel"] = ak.zip({"pt": loose, "eta": raw.Jet.eta})
