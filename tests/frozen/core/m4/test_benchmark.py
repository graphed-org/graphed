"""The super-linear guard: reduction wall time must NOT grow super-linearly with graph size across
{1k, 2k, 4k, 8k} — the single most important guard against re-introducing dask's O(N^2) (plan M4)."""

from __future__ import annotations

import time

import graphed_core as gc

SIZES = [1000, 2000, 4000, 8000]


def _build(n: int) -> tuple[gc.GraphStore, int]:
    # a shared selection + many fused variations: a realistically-shaped graph of ~n nodes
    s = gc.GraphStore()
    src = s.add_source("events")
    sel = src
    for i in range(n // 4):
        sel = s.add_op("select", [sel], {"step": i})
    acc = sel
    for v in range(n - n // 4 - 1):
        leaf = s.add_op("observable", [sel], {"var": v})
        acc = s.add_op("add", [acc, leaf])
    return s, acc


def _time_reduce(store: gc.GraphStore, out: int, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        store.reduce(outputs=[out])
        best = min(best, time.perf_counter() - start)
    return best


def test_reduction_is_subquadratic() -> None:
    stores = {n: _build(n) for n in SIZES}
    times = {n: _time_reduce(*stores[n]) for n in SIZES}

    base = max(times[SIZES[0]], 1e-4)  # floor to avoid divide-by-noise on tiny times
    growth = times[SIZES[-1]] / base  # size grows 8x from 1k -> 8k
    # linear => ~8x; quadratic => ~64x. Fail well below quadratic; allow generous overhead.
    assert growth < 24.0, f"reduction scaling looks super-linear: 8x size -> {growth:.1f}x time {times}"


def test_each_size_reduces_within_budget() -> None:
    for n in SIZES:
        s, out = _build(n)
        start = time.perf_counter()
        _, report = s.reduce(outputs=[out])
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"n={n} took {elapsed:.3f}s"
        assert report["reduced_nodes"] < 10  # collapses to O(stage) regardless of n
