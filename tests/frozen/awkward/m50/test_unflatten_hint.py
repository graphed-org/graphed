"""§6.1d's broadcast refusal must carry BOTH halves: the offending factor AND the unflatten hint.

m49's frozen blame anchor (`awkward/m49/test_broadcast_blame.py`) asserts the factor half only, so an
implementation that drops the "pass the value unflattened" hint ships green under it. This carryover
freezes the hint half. The hint is CONDITIONAL — it fires on an already-flattened per-object value
broadcast against a per-event factor (both leaf structures), and stays OFF a mismatch between two
structured operands, where flattening is not the fault — so a translation that appends it
unconditionally is caught by the second member.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedError

#: two events; jagged per-object `Jet.pt`, per-event scalar `w`
EVENTS = ak.Array([{"Jet": [{"pt": 30.0}, {"pt": 12.0}], "w": 1.5}, {"Jet": [{"pt": 40.0}], "w": 0.5}])

#: the hint half — distinctive to the "pass it unflattened" clause, absent from the factor-naming
#: leading clause and from awkward's underlying complaint (see the module's captured messages)
UNFLATTEN_HINT = "unflatten"


def _blame(value: Any, factor: Any) -> str:
    session = Session(AwkwardBackend())
    broadcast = graphed.broadcast_like(
        from_awkward(session, "value", ak.Array(value)), from_awkward(session, "factor", ak.Array(factor))
    )
    with pytest.raises(GraphedError) as excinfo:
        session.materialize(broadcast)
    return str(excinfo.value)


def test_flattened_per_object_value_names_the_factor_and_the_unflatten_hint() -> None:
    # 3 flattened objects against 2 per-event factors: the case §6.1d's hint exists for
    message = _blame(ak.flatten(EVENTS.Jet.pt), EVENTS.w)
    assert "factor" in message  # m49 already guards this half
    assert UNFLATTEN_HINT in message.lower()  # this carryover guards the hint half


def test_two_structured_operands_are_blamed_without_the_unflatten_hint() -> None:
    counts = ak.Array([[1.0], [2.0, 3.0]])  # jagged; every row disagrees with Jet.pt, flattening is not the fix
    message = _blame(EVENTS.Jet.pt, counts)
    assert "factor" in message
    assert UNFLATTEN_HINT not in message.lower()
