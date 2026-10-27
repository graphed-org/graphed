"""Reducing an already-reduced store is a fixed point: distinct same-size stages stay distinct."""

from __future__ import annotations

import graphed.core as gc


def test_rereduce_keeps_same_size_stages_and_outputs() -> None:
    s = gc.GraphStore()
    e = s.add_source("e")
    a, b = s.add_op("neg", [e]), s.add_op("abs", [e])
    once, _ = s.reduce(outputs=[a, b])
    twice, _ = once.reduce()
    assert (once.node_count(), once.outputs()) == (3, [1, 2])
    assert (twice.node_count(), twice.outputs()) == (3, [1, 2])
    assert twice.serialize() == once.serialize()
