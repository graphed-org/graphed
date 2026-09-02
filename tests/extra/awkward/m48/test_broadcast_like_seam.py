"""§6.1d's `broadcast_like` seam under the awkward idiom.

The neutral verb's no-op arm is covered awkward-free in `tests/extra/frontend/m48`; this is the
recording arm — the one that makes a per-event factor usable against a jagged value.
"""

from __future__ import annotations

import awkward as ak

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward

EVENTS = ak.Array([{"Jet": [{"pt": 30.0}, {"pt": 12.0}], "w": 1.5}, {"Jet": [{"pt": 40.0}], "w": 0.5}])


def _session() -> tuple[Session, graphed.Array]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", EVENTS)


def test_a_per_event_factor_takes_the_jagged_values_structure() -> None:
    session, root = _session()
    jagged, per_event = root.Jet.pt, root.w

    spread = graphed.broadcast_like(jagged, per_event)
    assert spread.node_id != per_event.node_id  # a node was RECORDED, not the identity fallback
    assert str(session.form(spread)) == str(session.form(jagged))
    assert ak.to_list(session.materialize(spread)) == [[1.5, 1.5], [0.5]]


def test_the_seam_maps_over_the_universes_of_a_varied_factor() -> None:
    session, root = _session()
    jagged = root.Jet.pt
    factor = graphed.vary(root.w, "pu", up=root.w * 2.0)

    spread = graphed.broadcast_like(jagged, factor)
    assert isinstance(spread, graphed.Varied)
    assert list(graphed.labels(spread)) == ["nominal", "pu_up"]
    assert ak.to_list(session.materialize(graphed.universe(spread, "pu_up"))) == [[3.0, 3.0], [1.0]]
