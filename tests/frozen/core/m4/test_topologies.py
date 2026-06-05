"""Complex graph topologies through the optimizer (plan M4) — diamond, star (fan-out/fan-in), and
deeply-nested shapes. These were the dask-optimizer failure points (CSE blow-ups, O(N^2) reduction,
DCE dropping a node reachable via two paths), so they are guarded explicitly here: correct reduced
structure, a shared apex/hub interned once, and sub-quadratic reduction time as the shape grows.
"""

from __future__ import annotations

import time

import graphed_core as gc


def _diamond(s: gc.GraphStore, apex: int) -> int:
    """apex fans out to two distinct branches that re-converge; returns the join node."""
    left = s.add_op("incl", [apex])
    right = s.add_op("negr", [apex])
    return s.add_op("add", [left, right])


def test_diamond_apex_is_shared_and_not_duplicated() -> None:
    s = gc.GraphStore()
    x = s.add_source("x")
    apex = s.add_op("inc", [x])  # out-degree 2
    out = _diamond(s, apex)
    s.mark_output(out)
    assert s.node_count() == 5  # x, apex, left, right, join — apex interned ONCE
    _reduced, report = s.reduce()
    # the fan-out apex stays its own stage (never duplicated into both branches)
    assert report["stages"] == 2
    assert report["reduced_nodes"] == 3  # source + apex-stage + branch-stage


def test_star_fan_out_then_fan_in() -> None:
    n = 32
    s = gc.GraphStore()
    x = s.add_source("x")
    hub = s.add_op("inc", [x])  # fans out to n consumers
    leaves = [s.add_op("add", [hub, s.add_source(f"s{i}")]) for i in range(n)]
    out = s.add_op("add", leaves)  # fan-in, in-degree n
    s.mark_output(out)
    _reduced, report = s.reduce()
    # the hub is one shared node; the leaves + fan-in fuse — O(stage), not O(n) stages
    assert report["stages"] <= 3
    assert report["reachable_nodes"] == s.node_count()  # nothing dropped


def test_nested_stacked_diamonds_reduce_linearly() -> None:
    for d in (1, 8, 64, 256):
        s = gc.GraphStore()
        v = s.add_source("x")
        for _ in range(d):
            v = _diamond(s, v)  # each apex is a fan-out
        s.mark_output(v)
        _reduced, report = s.reduce()
        # reduced size grows at most linearly with depth (no exponential / quadratic blow-up)
        assert report["reduced_nodes"] <= 3 * d + 2, f"d={d}"


def test_dead_branch_off_a_diamond_is_dropped_but_diamond_kept() -> None:
    s = gc.GraphStore()
    x = s.add_source("x")
    apex = s.add_op("inc", [x])
    out = _diamond(s, apex)
    s.add_op("mul", [apex, x])  # a dead branch off the apex (never an output)
    s.mark_output(out)
    _reduced, report = s.reduce()
    # the apex is reachable via both diamond branches and survives; only the dead mul is dropped
    assert report["reachable_nodes"] == s.node_count() - 1


def test_topology_reduction_is_deterministic() -> None:
    def build() -> str:
        s = gc.GraphStore()
        a, b = s.add_source("a"), s.add_source("b")
        hub = s.add_op("add", [a, b])
        out = s.add_op("mul", [_diamond(s, hub), s.add_op("inc", [hub])])
        s.mark_output(out)
        return s.reduce()[0].to_dot()

    assert build() == build()


def _diamond_chain(nodes: int) -> gc.GraphStore:
    # ~3 nodes per stacked diamond
    s = gc.GraphStore()
    v = s.add_source("x")
    for _ in range(max(1, nodes // 3)):
        v = _diamond(s, v)
    s.mark_output(v)
    return s


def _star(nodes: int) -> gc.GraphStore:
    # a real growing star: each consumer takes a DISTINCT source so the leaves don't intern to one
    # (~2 nodes per star point). Hub fans out; leaves fan in.
    s = gc.GraphStore()
    hub = s.add_op("inc", [s.add_source("x")])
    leaves = [s.add_op("add", [hub, s.add_source(f"s{i}")]) for i in range(max(1, nodes // 2))]
    s.mark_output(s.add_op("add", leaves))
    return s


def _best_reduce_time(store: gc.GraphStore, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        store.reduce()
        best = min(best, time.perf_counter() - t)
    return best


def test_diamond_and_star_reduction_is_subquadratic() -> None:
    # the dask O(N^2) guard, on topologies (not just chains): an 8x bigger graph must NOT cost ~64x
    # the reduction time. Uses ms-scale sizes (matching the M4 benchmark) so the ratio is meaningful.
    for family in (_diamond_chain, _star):
        sizes = [1000, 2000, 4000, 8000]
        stores = {k: family(k) for k in sizes}
        times = {k: _best_reduce_time(stores[k]) for k in sizes}
        base = max(times[sizes[0]], 1e-3)  # floor at 1ms so timer noise on a fast base can't dominate
        growth = times[sizes[-1]] / base  # size grows 8x; linear ~8x, quadratic ~64x
        # these shapes (n-ary fan-in, deep nesting) carry a higher constant than a clean chain; 30x
        # still cleanly separates sub-linear/linear-ish from the 64x quadratic this guards against.
        assert growth < 30.0, f"{family.__name__}: 8x size -> {growth:.1f}x time {times}"
