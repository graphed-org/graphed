"""M35 — Session.walk evaluates iteratively: a deeply-chained graph does not hit the recursion limit.

`materialize` and projection share `Session.walk`. A long recorded chain (a realistic deep
selection/systematics graph) would `RecursionError` if `walk` recursed per node. The traversal is
now explicit-stack, so depth far beyond Python's recursion limit materializes correctly and in
left-to-right order.
"""

from __future__ import annotations

import sys

import numpy as np
from graphed_numpy import NumpyBackend, from_array

from graphed import Session


def test_deep_chain_materializes_past_the_recursion_limit() -> None:
    depth = sys.getrecursionlimit() * 3  # well beyond what a per-node recursion could survive
    s = Session(NumpyBackend())
    x = from_array(s, "x", np.zeros(4))
    acc = x
    for _ in range(depth):
        acc = acc + 1.0
    got = np.asarray(s.materialize(acc))
    assert np.array_equal(got, np.full(4, float(depth)))  # correct value, no RecursionError


def test_shared_subgraph_evaluated_once_in_a_deep_graph() -> None:
    s = Session(NumpyBackend())
    x = from_array(s, "x", np.arange(5.0))
    shared = x * 2.0
    a = shared + 1.0
    b = shared + 2.0
    out = a + b  # diamond: `shared` reached by two paths
    assert np.array_equal(np.asarray(s.materialize(out)), np.arange(5.0) * 4.0 + 3.0)
