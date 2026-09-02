"""§2.5's unreached-label scan reaches DOWNSTREAM of a container.

The label rides the frontend proxy, not the IR (§1.2), so a marked output derived several ops past
a varied member still has to carry it — a scan that only looks at the member itself reports a
systematic that is being filled as unreached.
"""

from __future__ import annotations

import numpy as np
from m49_vary_fixtures import VECTOR

import graphed
from graphed import Session, compile_ir
from graphed.numpy import NumpyBackend, from_array

DOWNSTREAM_OPS = 4


def _program() -> tuple[Session, graphed.Array, graphed.Array]:
    session = Session(NumpyBackend())
    x = from_array(session, "x", VECTOR)
    reached = graphed.vary(x, "jes", up=x * 1.1)
    graphed.vary(x, "btag", up=x * 2.0)  # registered, never marked: the silent-cost case
    deep = graphed.universe(reached, "jes_up")
    for step in range(DOWNSTREAM_OPS):
        deep = deep + float(step + 1)
    return session, graphed.universe(reached, "nominal").sum(), deep.sum()


def test_a_label_reached_only_downstream_is_not_reported_unreached() -> None:
    session, central, deep = _program()
    compiled = compile_ir(session, central, deep)

    assert "jes_up" not in compiled.unreached_labels


def test_a_registered_label_no_output_reaches_is_still_reported() -> None:
    """The control leg in the same program: the scan is live, so the first assertion means
    something."""
    session, central, deep = _program()
    compiled = compile_ir(session, central, deep)

    assert compiled.unreached_labels == ("btag_up",)


def test_the_downstream_output_still_evaluates_in_the_shifted_universe() -> None:
    session, _central, deep = _program()
    expected = float((VECTOR * 1.1 + sum(float(s + 1) for s in range(DOWNSTREAM_OPS))).sum())
    assert np.isclose(float(np.asarray(session.materialize(deep))), expected)
