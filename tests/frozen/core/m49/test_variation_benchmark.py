"""The anti-quadratic guard on the variation topology: a shared prefix feeding N per-universe
chains, each terminating in its own separately-marked reduction, must reduce to an exactly pinned
shape in time that grows linearly with N (plan §3.3). The m4 files are frozen and untouched."""

from __future__ import annotations

import time
from collections import Counter

import graphed.core as gc

SHARED_PREFIX_DEPTH = 500  # D
UNIVERSE_CHAIN_LENGTH = 50  # K
UNIVERSE_COUNTS = [16, 32, 64, 128]  # N, counting nominal


def _build(n_universes: int) -> tuple[gc.GraphStore, list[int]]:
    """source -> D shared-prefix ops -> per universe {one varied fork op, K chain ops, exactly one
    terminating reduction}, every universe's reduction separately marked as an output.

    Funnelling the universes into one output (m4's `_systematics` shape) or dropping the
    terminating reduction both change the correct answer; the shape assertion pins this builder.
    """
    store = gc.GraphStore()
    node = store.add_source("events", {"uri": "f.root"})
    for step in range(SHARED_PREFIX_DEPTH):
        node = store.add_op("select", [node], {"step": step})
    outputs = []
    for universe in range(n_universes):
        label = "nominal" if universe == 0 else f"shift_{universe}"
        chain = store.add_op("shift", [node], {"label": label})
        for k in range(UNIVERSE_CHAIN_LENGTH):
            chain = store.add_op("chain", [chain], {"k": k, "label": label})
        outputs.append(store.add_reduction("hist", [chain], {"label": label}))
    return store, outputs


def _time_reduce(store: gc.GraphStore, outputs: list[int], repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        store.reduce(outputs=outputs)
        best = min(best, time.perf_counter() - start)
    return best


def test_variation_topology_reduces_to_the_pinned_shape() -> None:
    for n in UNIVERSE_COUNTS:
        store, outputs = _build(n)
        reduced, report = store.reduce(outputs=outputs)

        assert report["stages"] == n + 1, f"N={n}: stages {report['stages']} != {n + 1}"
        assert report["reduced_nodes"] == 2 * n + 2, (
            f"N={n}: reduced_nodes {report['reduced_nodes']} != {2 * n + 2}"
        )
        # the reduced store, not the report, says what the shape is made of
        kinds = Counter(node["kind"] for node in reduced.nodes())
        assert kinds == Counter({"stage": n + 1, "reduction": n, "source": 1}), f"N={n}: {kinds}"
        assert len(reduced.outputs()) == n, f"N={n}: {len(reduced.outputs())} marked outputs"


def test_reduction_time_grows_linearly_in_the_universe_count() -> None:
    stores = {n: _build(n) for n in UNIVERSE_COUNTS}
    times = {n: _time_reduce(*stores[n]) for n in UNIVERSE_COUNTS}

    base = max(times[UNIVERSE_COUNTS[0]], 1e-4)  # floor to avoid divide-by-noise on tiny times
    growth = times[UNIVERSE_COUNTS[-1]] / base
    # node count grows 5.37x while N grows 8x, so a node-quadratic reducer lands near 29x.
    assert growth < 16.0, f"reduction scaling looks super-linear: 8x N -> {growth:.1f}x time {times}"
