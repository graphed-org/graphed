"""The systematics-graph headline: a bloated graph with heavy shared substructure reduces to a
concise stage-graph whose size scales with stages, not with op/systematics count (plan M4)."""

from __future__ import annotations

import time

import graphed.core as gc


def _systematics(n_variations: int, selection_depth: int) -> tuple[gc.GraphStore, int]:
    """A shared selection chain feeding many per-variation observable+weight tails that combine
    into one output — the structure the AGC systematics graph has (heavy shared substructure)."""
    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "f.root"})
    sel = src
    for i in range(selection_depth):
        sel = s.add_op("select", [sel], {"step": i})  # shared selection (computed once)
    acc: int | None = None
    for v in range(n_variations):
        obs = s.add_op("observable", [sel], {"var": v})  # reads the shared selection (fan-out)
        w = s.add_op("weight", [obs], {"var": v})
        acc = w if acc is None else s.add_op("add", [acc, w])
    assert acc is not None
    return s, acc


def test_reduced_size_is_independent_of_systematics_count() -> None:
    # doubling (10x-ing) the variation count must NOT grow the reduced stage-graph: the variation
    # ops all fuse into one stage; the shared selection stays one stage.
    s_small, out_small = _systematics(50, 30)
    s_big, out_big = _systematics(500, 30)
    small, _ = s_small.reduce(outputs=[out_small])
    big, _ = s_big.reduce(outputs=[out_big])
    assert small.node_count() == big.node_count()
    assert small.node_count() <= 5  # source + selection-stage + variation-stage


def test_ten_thousand_node_graph_reduces_fast_to_o_stage() -> None:
    s, out = _systematics(n_variations=3300, selection_depth=100)  # ~10,000 nodes
    n_in = s.node_count()
    assert n_in >= 9000

    start = time.perf_counter()
    reduced, report = s.reduce(outputs=[out])
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"reduction took {elapsed:.3f}s (budget 1s)"
    # O(stage): the un-reduced O(10^4) graph collapses to a handful of nodes
    assert reduced.node_count() <= 8
    assert report["reachable_nodes"] == n_in  # everything is live
    # the headline win vs the graph-bloat note: a >1000x node-count reduction
    assert report["reduced_nodes"] * 1000 < report["reachable_nodes"]
