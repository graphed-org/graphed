"""M40 — the numpy ``ShuffleBackend`` *join* half over structured (record) arrays.

M39 shipped the exchange half (``partition``/``concat``/``slice_rows``/``to_wire``/…); M40 adds the
relational half — ``match_indices`` / ``take`` / ``merge_records`` — that turns two co-partitioned
record blocks into the SQL join result. The pin (spec §INVARIANTS, plan §3.3): a probe row with ``k``
matches yields ``k`` output rows (relational duplication), NOT one list-of-matches / grouped row.
numpy's ``match_indices`` is argsort+searchsorted (never Python ``hash()``), so it must be correct on
UNSORTED input with duplicate keys — this file witnesses exactly that on tiny synthetic blocks (the
corpus-skim bit-for-bit test lives in ``frozen/frontend/m40``).
"""

from __future__ import annotations

import numpy as np

from graphed.numpy import NumpyBackend, shuffle

_KEY = "__joinkey__"


def _rec(keys: list[int], **cols: object) -> np.ndarray:
    """A record block: the u64 ``__joinkey__`` column plus named data columns."""
    dt = [(_KEY, np.uint64), *[(k, np.asarray(v).dtype) for k, v in cols.items()]]
    a = np.zeros(len(keys), dtype=dt)
    a[_KEY] = np.asarray(keys, dtype=np.uint64)
    for k, v in cols.items():
        a[k] = np.asarray(v)
    return a


def _pairs(build_idx: np.ndarray, probe_idx: np.ndarray) -> set[tuple[int, int]]:
    return set(zip((int(i) for i in build_idx), (int(j) for j in probe_idx), strict=True))


def test_match_indices_is_relational_on_unsorted_duplicate_keys() -> None:
    # (a) THE relational-duplication pin: build key 20 appears at rows 0 AND 2 (unsorted, duplicated);
    # probe key 20 at row 0 ⇒ the inner join must emit BOTH pairs (0,0) and (2,0) — two rows, not one.
    be = NumpyBackend()
    build = _rec([20, 10, 20, 30], v=np.arange(4))
    probe = _rec([20, 40], w=np.arange(2))
    build_idx, probe_idx = be.match_indices(build, probe, on=[_KEY], how="inner")
    assert len(build_idx) == len(probe_idx) == 2  # k matches ⇒ k rows (grouped baseline gives 1)
    assert _pairs(build_idx, probe_idx) == {(0, 0), (2, 0)}  # correct on UNSORTED input


def test_match_indices_inner_drops_unmatched() -> None:
    # inner keeps only matched pairs; keys 10/30 (build-only) and 40 (probe-only) vanish.
    be = NumpyBackend()
    build = _rec([10, 20, 30], v=np.arange(3))
    probe = _rec([20], w=np.arange(1))
    build_idx, probe_idx = be.match_indices(build, probe, on=[_KEY], how="inner")
    assert _pairs(build_idx, probe_idx) == {(1, 0)}


def test_take_gathers_records_by_index() -> None:
    # ``take`` is a positional gather over the record axis (the join then feeds it match indices).
    be = NumpyBackend()
    block = _rec([10, 20, 30, 40], v=[0, 1, 2, 3])
    out = be.take(block, np.array([2, 0, 3]))
    assert list(np.asarray(out[_KEY])) == [30, 10, 40]
    assert list(np.asarray(out["v"])) == [2, 0, 3]


def test_merge_records_is_union_of_fields_minus_duplicate_key() -> None:
    # ``merge_records`` combines two row-aligned blocks: __joinkey__ once, both data columns kept.
    be = NumpyBackend()
    left = _rec([10, 20], lx=np.array([100, 200], dtype=np.int64))
    right = _rec([10, 20], rx=np.array([1.5, 2.5], dtype=np.float64))
    merged = be.merge_records(left, right, on=[_KEY])
    names = [f for f, _ in merged.dtype.descr]
    assert names.count(_KEY) == 1  # the shared key is NOT duplicated
    assert {"lx", "rx"} <= set(names)
    assert list(np.asarray(merged["lx"])) == [100, 200]
    assert list(np.asarray(merged["rx"])) == [1.5, 2.5]


def _inner_join(be: NumpyBackend, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The generic kernel composed from the numpy primitives (§B-14 shape)."""
    b, p = be.match_indices(left, right, on=[_KEY], how="inner")
    return be.merge_records(be.take(left, b), be.take(right, p), on=[_KEY])


def test_inner_join_matches_a_duplicating_sql_baseline() -> None:
    # (a) bit-for-bit relational content vs an INDEPENDENT duplicating (SQL-style) oracle. Compared as
    # a multiset so the test does not pin an arbitrary output order; duplication (k rows) is the point.
    be = NumpyBackend()
    lk, lv = [10, 20, 20, 30], [1, 2, 3, 4]
    rk, rv = [20, 20, 40], [5.0, 6.0, 7.0]
    left = _rec(lk, lx=np.array(lv, dtype=np.int64))
    right = _rec(rk, rx=np.array(rv, dtype=np.float64))
    expected = sorted(
        (int(a), int(x), float(y))
        for a, x in zip(lk, lv, strict=True)
        for b, y in zip(rk, rv, strict=True)
        if a == b
    )  # key 20: 2 left x 2 right = 4 rows — a grouped/list-of-matches join would give far fewer
    merged = _inner_join(be, left, right)
    got = sorted(
        (int(k), int(x), float(y)) for k, x, y in zip(merged[_KEY], merged["lx"], merged["rx"], strict=True)
    )
    assert got == expected
    assert len(got) == 4


def test_inner_join_is_deterministic() -> None:
    # identical inputs ⇒ byte-identical output across two runs (the determinism gate for the join half).
    be = NumpyBackend()
    left = _rec([10, 20, 20, 30], lx=np.arange(4, dtype=np.int64))
    right = _rec([20, 40, 20], rx=np.arange(3, dtype=np.float64))
    r1 = _inner_join(be, left, right)
    r2 = _inner_join(be, left, right)
    assert r1.tobytes() == r2.tobytes()


def test_join_free_functions_exist_on_the_shuffle_module() -> None:
    # §B-6: the join primitives are pure free fns in graphed.numpy.shuffle (the backend delegates to
    # them; the generic engine imports them). Witness they exist AND agree with the delegate.
    left = _rec([10, 20], lx=np.array([1, 2], dtype=np.int64))
    right = _rec([20, 30], rx=np.array([3.0, 4.0], dtype=np.float64))
    b, p = shuffle.match_indices(left, right, on=[_KEY], how="inner")
    merged = shuffle.merge_records(shuffle.take(left, b), shuffle.take(right, p), on=[_KEY])
    assert list(np.asarray(merged["lx"])) == [2] and list(np.asarray(merged["rx"])) == [3.0]
