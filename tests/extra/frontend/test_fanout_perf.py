"""Perf regression guards for systematics fanout CONSTRUCTION (the graphed.vary path).

These pin the SCALING SHAPE the perf/systematics-fanout work established, so a reintroduced
quadratic (or a dropped memo) is caught even though it breaks no correctness assertion -- the frozen
suites verify the answers a fanout produces, not what they cost. Numpy-idiom only (no awkward), so
the guard stays cheap and portable across the CI matrix.

Two independent pathologies are pinned:
  * vary._check_unique's Session point-uniqueness test must stay ~linear in accumulated universes;
    it rescanned the whole point registry per minted label -> O(N^2) before the reverse index.
  * vary._source_ids must resolve a shared prefix ONCE across a family's members, not once per
    member; the per-node memo on the Session is the mechanism, witnessed directly below.
"""

from __future__ import annotations

import time

import numpy as np

import graphed
import graphed.provenance as provenance
from graphed import Session
from graphed.numpy import NumpyBackend, from_record

VEC = np.arange(1.0, 13.0)


def _register_independent_families(n: int) -> None:
    """n independent 2-universe families on one base array in one Session: each vary() mints into
    the shared point registry, so _check_unique is exercised against a registry that grows with n."""
    session = Session(NumpyBackend())
    pt = from_record(session, "ev", pt=VEC)["pt"]
    for i in range(n):
        graphed.vary(pt, f"f{i}", up=pt * (1.0 + 1e-3 * (i + 1)), down=pt * (1.0 - 1e-3 * (i + 1)))


def _time_build(n: int, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        _register_independent_families(n)
        best = min(best, time.perf_counter() - start)
    return best


def test_registry_accumulation_is_not_quadratic() -> None:
    # Provenance is a fixed per-op cost that cancels in the ratio; hold it off so the ratio reflects
    # the registry-uniqueness term alone.
    was = provenance.is_enabled()
    provenance.set_enabled(False)
    try:
        small = max(_time_build(100), 1e-4)  # floor: avoid divide-by-noise on a sub-millisecond build
        large = _time_build(800)
    finally:
        provenance.set_enabled(was)
    growth = large / small
    # Families grow 8x. Near-linear construction lands ~10x (a small residual super-linearity in the
    # per-call minted scan); the pre-fix O(N^2) registry rescan lands ~30x. A bound between the two
    # catches a reintroduced quadratic without tripping on timing noise.
    assert growth < 20.0, f"registry-accumulation scaling looks super-linear: 8x families -> {growth:.1f}x time"


def test_source_ids_memoizes_the_shared_prefix() -> None:
    # A multi-universe vary over a deep prefix routes every member through check_members ->
    # _source_ids. With the per-node memo the shared prefix is walked once and every reachable node
    # is cached exactly once; drop the memo and the cache stays empty -- so this both witnesses the
    # mechanism and fails on its removal.
    session = Session(NumpyBackend())
    pt = from_record(session, "ev", pt=VEC)["pt"]
    for _ in range(200):  # a deep shared prefix all three universes (nominal/up/down) read through
        pt = pt * 1.0001
    graphed.vary(pt, "sys", up=pt * 1.1, down=pt * 0.9)  # loose form -> check_members -> _source_ids
    cache = session._source_ids_cache
    assert cache, "the _source_ids memo is empty after a multi-universe vary over a deep prefix"
    # the whole deep prefix is memoized (one entry per reachable node), not re-walked per member
    assert len(cache) >= 200, f"deep prefix not fully memoized: cache holds {len(cache)} entries"
