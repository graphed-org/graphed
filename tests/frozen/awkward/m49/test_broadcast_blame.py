"""§6.1d's broadcast seam has to blame the factor.

A weight factor whose structure disagrees with the value it is broadcast against, while AGREEING on
outer length, gets past the histogram-side guard — that guard compares row counts only. What the user
then sees today is awkward's own complaint about two anonymous layouts, which names neither operand
and points at neither. The seam's awkward implementation must translate it into a `graphed` error
naming the offending FACTOR.

The two members bracket the class, and they fail on different paths: a regular-vs-regular mismatch is
decidable from the typetracer forms and lands at RECORD time, while a jagged pair whose counts differ
only in the data lands at EXECUTION time. An implementation that repairs one path leaves the other.
The compatible-factor positive control rides each case, compared against a manually broadcast
reference — a translation that fires on every mismatch AND on every match witnesses nothing.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest
from m49_ctx_fixtures import (
    EVENTS,
    JAGGED_FACTOR,
    REGULAR_COMPATIBLE,
    REGULAR_FACTOR,
    REGULAR_VALUE,
    as_list,
    inner_type,
)

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import GraphedError

#: (value, offending factor, compatible factor), all agreeing on outer length
CASES = {
    "regular": (REGULAR_VALUE, REGULAR_FACTOR, REGULAR_COMPATIBLE),
    "jagged": (EVENTS.Jet.pt, JAGGED_FACTOR, EVENTS.MET.pt),
}


def _read(session: Session, name: str, data: Any) -> Any:
    return from_awkward(session, name, ak.Array(data))


@pytest.mark.parametrize("case", list(CASES))
def test_a_structure_mismatch_agreeing_on_outer_length_blames_the_factor(case: str) -> None:
    value_data, bad_data, good_data = CASES[case]
    assert len(value_data) == len(bad_data) == len(good_data), "the row-count guard cannot fire here"
    session = Session(AwkwardBackend())
    value = _read(session, "value", value_data)

    broadcast = graphed.broadcast_like(value, _read(session, "good", good_data))
    assert as_list(session.materialize(broadcast)) == as_list(ak.broadcast_arrays(value_data, good_data)[1])

    with pytest.raises(GraphedError) as excinfo:
        session.materialize(graphed.broadcast_like(value, _read(session, "bad", bad_data)))

    message = str(excinfo.value)
    assert "factor" in message
    assert inner_type(bad_data) in message
