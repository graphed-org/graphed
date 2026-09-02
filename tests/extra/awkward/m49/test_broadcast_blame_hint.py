"""§6.1d's blame message points at "pass the value unflattened" only where that IS the fault.

The frozen anchor pins the class (a `graphed` error naming the factor); this pins the hint's
discrimination — it fires on the already-flattened per-object value, and stays off a mismatch
between two structured operands, where flattening is not the problem.
"""

from __future__ import annotations

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedError

EVENTS = ak.Array([{"Jet": [{"pt": 30.0}, {"pt": 12.0}], "w": 1.5}, {"Jet": [{"pt": 40.0}], "w": 0.5}])
UNFLATTEN = "UNFLATTENED"


def _blame(value: ak.Array, factor: ak.Array) -> str:
    session = Session(AwkwardBackend())
    broadcast = graphed.broadcast_like(
        from_awkward(session, "value", value), from_awkward(session, "factor", factor)
    )
    with pytest.raises(GraphedError) as excinfo:
        session.materialize(broadcast)
    return str(excinfo.value)


def test_a_flattened_per_object_value_is_told_to_pass_it_unflattened() -> None:
    message = _blame(ak.flatten(EVENTS.Jet.pt), EVENTS.w)  # 3 objects against 2 events
    assert "factor" in message
    assert UNFLATTEN in message


def test_two_structured_operands_get_the_blame_without_the_flatten_hint() -> None:
    counts = ak.Array([[1.0], [2.0, 3.0]])  # jagged, and every row disagrees with Jet.pt
    message = _blame(EVENTS.Jet.pt, counts)
    assert "factor" in message
    assert UNFLATTEN not in message
