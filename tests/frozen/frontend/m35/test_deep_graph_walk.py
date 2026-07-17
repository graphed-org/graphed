"""M35 — Session.walk evaluates iteratively: a deeply-chained graph does not hit the recursion limit.

`materialize` and projection share `Session.walk`. A long recorded chain (a realistic deep
selection/systematics graph) would `RecursionError` if `walk` recursed per node. The traversal is
now explicit-stack, so depth far beyond Python's recursion limit materializes correctly. Uses the
backend-independent ListBackend (graphed depends only on graphed-core; no array backend installed).
"""

from __future__ import annotations

import sys

from backends import ListBackend, from_list

from graphed import Session


def test_deep_chain_materializes_past_the_recursion_limit() -> None:
    depth = sys.getrecursionlimit() * 3  # well beyond what a per-node recursion could survive
    s = Session(ListBackend())
    base = from_list(s, "one", [1, 1, 1, 1])
    acc = from_list(s, "x", [0, 0, 0, 0])
    for _ in range(depth):
        acc = acc + base  # each add deepens the chain by one node
    assert s.materialize(acc) == [depth, depth, depth, depth]  # correct value, no RecursionError


def test_shared_subgraph_evaluated_correctly_in_a_diamond() -> None:
    s = Session(ListBackend())
    x = from_list(s, "x", [1, 1, 1, 1])
    y = from_list(s, "y", [2, 2, 2, 2])
    shared = x + y  # reached by both paths below
    out = (shared + x) + (shared + y)  # diamond over `shared`
    assert s.materialize(out) == [9, 9, 9, 9]  # (3+1) + (3+2), elementwise
