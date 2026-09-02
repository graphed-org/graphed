"""§1.1's family tags survive a §2.4 combining op.

A combined container advertises the union of its operands' labels; if it drops the per-family TAG
map with them, the next `vary` on that result sees an empty family and admits a second spelling of
a universe it already carries. The label check alone does not catch it: `5em1` and `0p5` are
distinct labels naming one value.
"""

from __future__ import annotations

import numpy as np
import pytest
from m49_vary_fixtures import VECTOR

import graphed
from graphed import Session
from graphed.errors import GraphedError
from graphed.numpy import NumpyBackend, from_array

HALF = 0.5


def _combined() -> tuple[Session, graphed.Array, graphed.Varied]:
    session = Session(NumpyBackend())
    x = from_array(session, "x", VECTOR)
    scale = graphed.vary(x, "scale", **{"0.5": x * HALF})
    other = graphed.vary(x, "btag", up=x * 2.0)
    return session, x, scale + other


def test_a_combining_op_unions_the_labels() -> None:
    _session, _x, combined = _combined()
    assert graphed.labels(combined) == ("nominal", "scale_5em1", "btag_up")


def test_a_family_tag_survives_the_combination() -> None:
    """`0p5` canonicalizes to a DIFFERENT label than `5em1` but names the same value, so only the
    family check can refuse it — and only if the combination carried the family through."""
    _session, x, combined = _combined()

    with pytest.raises(GraphedError) as caught:
        graphed.vary(combined, "scale", **{"0p5": x * HALF})
    message = str(caught.value)
    assert "scale" in message
    assert "0p5" in message


def test_a_genuinely_new_tag_in_that_family_is_still_accepted() -> None:
    """The control leg: the family check refuses the duplicate value, not every later `vary`."""
    session, x, combined = _combined()
    grown = graphed.vary(combined, "scale", **{"2.0": x * 2.0})

    assert graphed.labels(grown) == ("nominal", "scale_5em1", "btag_up", "scale_2")
    assert np.allclose(
        np.asarray(session.materialize(graphed.universe(grown, "scale_2"))), VECTOR * 2.0
    )
