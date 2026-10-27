"""A `graphed.weight()` read at a `vary` child names its composition at the parent too.

A `vary` link keeps the row space, so the two share their read records.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import graphed
from graphed import Session
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record


def test_a_handle_read_at_a_shift_child_anchors_an_overlay_registered_on_its_parent() -> None:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(1.0, 13.0))
    ctx = EventContext(session, record["pt"], collections={"pt": record["pt"]})
    ctx = graphed.vary(ctx, "a", ctx["pt"] * 0.5, is_weight=True, up=ctx["pt"] * 0.6)
    parent = graphed.vary(ctx, "b", ctx["pt"] * 0.7, is_weight=True, up=ctx["pt"] * 0.8)
    shifted = graphed.vary(
        parent, "s", collections={"pt": {"up": record["pt"] * 1.01, "down": record["pt"] * 0.99}}
    )
    handle: Any = graphed.weight(shifted)
    assert handle is not None

    registered = graphed.vary(parent, "c", handle, is_weight=True, up=handle * 1.1)

    composed = graphed.weight(registered)
    assert composed is not None
    assert graphed.nominal(composed).node_id == graphed.nominal(handle).node_id
    assert "an overlay over a, b" in str(graphed.explain(registered))


def test_a_widening_join_on_a_vary_child_leaves_the_parents_handle_fresh_there() -> None:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(1.0, 13.0))
    ctx = EventContext(session, record["pt"], collections={"pt": record["pt"]})
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {"up": record["pt"] * 1.01, "down": record["pt"] * 0.99}}
    )
    flat = graphed.nominal(shifted["pt"])
    parent = graphed.vary(shifted, "hf", flat * 0.5, is_weight=True, up=flat * 0.6)
    handle: Any = graphed.weight(parent)
    assert handle is not None
    widening = shifted["pt"] * 0.5
    graphed.vary(parent, "lf", widening, is_weight=True, up=widening * 1.2)

    registered = graphed.vary(parent, "mu", handle, is_weight=True, up=handle * 1.1)

    assert "mu (WEIGHT) ['up'] at the root: an overlay over hf" in str(graphed.explain(registered))
