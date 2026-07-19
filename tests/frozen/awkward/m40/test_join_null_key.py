"""M40 iteration-4 — a relational join ON an OPTION-typed key carrying a GENUINE null on a PRESENT,
UNMATCHED row must KEEP the row, PRESERVE the null key value, and keep the key OPTION-typed.

Plan §3.3 (SQL/pandas ``merge``): a NULL join key matches nothing, so a present row whose ``on`` key
is null survives the join as an *unmatched* row (other side null), with its null key intact. Option
``on`` keys are the norm in awkward (a column is option-typed after a cut/mask), so this is reachable.

On HEAD the awkward primitive ``merge_records`` coalesces the key with
``ak.drop_none(ak.where(ak.is_none(left[k]), right[k], left[k]))``. That conflates "this row's present
side is ABSENT (a take miss)" with "this row's key VALUE is null": for a present-but-unmatched row the
genuine null key is treated as a miss, so ``where`` swaps in the OTHER side (itself a miss ⇒ null) and
``drop_none`` then DROPS the row / mismatches the block length (crash) / corrupts a neighbour's key.
Measured on HEAD (see the per-test failure excerpts committed alongside this suite):
  * key ``[1,None,4]`` how=left/outer  -> RAISES ``ValueError: cannot broadcast ... size 2 with ... 3``
  * key ``[None,2]``   how=left        -> keys corrupt to ``[2,2]`` (the null row's key becomes 2)
  * key ``[None]``     how=left        -> total row loss (len 0)
The NUMPY backend handles the SAME masked-key input correctly, so awkward now DIVERGES from numpy —
breaking the M40 two-backend parity invariant on a reachable analysis shape.

Oracle independence (non-vacuity): every expected multiset is computed by ``pandas.merge`` (the SQL
reference used across the M40 frontend suite, independent of BOTH backends) — never read from the
awkward code under test. The numpy backend is asserted to reproduce the SAME multiset both as an
independent second oracle AND as the explicit parity peer. NOTE: no scenario places a null on BOTH
sides of a key, so pandas' NaN==NaN merge quirk cannot apply — a null here provably matches nothing.
Rows are compared as MULTISETS (``Counter``): awkward and numpy may emit rows in a different order.
"""

from __future__ import annotations

from collections import Counter

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from graphed.awkward import AwkwardBackend
from graphed.numpy import NumpyBackend

# Underlying int stored at a MASKED numpy key slot. numpy ``match_indices`` compares ``getdata``
# (ignoring the mask), so this must be ABSENT from the opposite side's keys for the SQL "null matches
# nothing" semantics to hold; every test key is a single digit, so this large constant cannot collide.
# (A real analysis's pre-mask value is arbitrary garbage; we pin a non-colliding one to model the SQL
# contract. awkward independently fills its own null slot with INT64_MAX — also non-colliding.)
_SENTINEL = 10_000_000

# (name, how, left_k, left_lv, right_k, right_rv, failure_mode) — left carries {k, lv}, right {k, rv};
# `None` in a key list is a GENUINE null on a present row. Modes: crash / total_loss / corruption.
_SCENARIOS = [
    ("crash_left_partial", "left", [1, None, 4], [10, 20, 30], [1, 4], [100, 400], "crash"),
    ("crash_outer_partial", "outer", [1, None], [10, 20], [1, 3], [100, 300], "crash"),
    ("total_loss_all_null", "left", [None], [20], [7], [700], "total_loss"),
    ("corrupt_left_one_null", "left", [None, 2], [20, 21], [2], [200], "corruption"),
    ("corrupt_right_one_null", "right", [5], [50], [None, 5], [600, 650], "corruption"),
]


def _masked(values: list[object]) -> np.ma.MaskedArray:
    """A 1-D masked int64 column; a ``None`` entry -> masked slot holding ``_SENTINEL``."""
    data = np.array([_SENTINEL if v is None else v for v in values], dtype=np.int64)
    mask = np.array([v is None for v in values], dtype=bool)
    return np.ma.MaskedArray(data, mask=mask)


def _ak_block(k: list[object], payload_name: str, payload: list[int]) -> ak.Array:
    """Awkward record block with an OPTION-typed ``k`` (``?int64``, a genuine null where ``k`` is None)."""
    return ak.Array({"k": ak.Array(_masked(k)), payload_name: np.asarray(payload, dtype=np.int64)})


def _np_block(k: list[object], payload_name: str, payload: list[int]) -> np.ma.MaskedArray:
    """Numpy structured masked block equivalent to :func:`_ak_block` (same logical table)."""
    km = _masked(k)
    dt = np.dtype([("k", np.int64), (payload_name, np.int64)])
    data = np.zeros(len(k), dtype=dt)
    mask = np.zeros(len(k), dtype=np.ma.make_mask_descr(dt))
    data["k"], mask["k"] = np.ma.getdata(km), np.ma.getmaskarray(km)
    data[payload_name] = np.asarray(payload, dtype=np.int64)
    return np.ma.MaskedArray(data, mask=mask)


def _triple(k: object, lv: object, rv: object) -> tuple[object, object, object]:
    return (k, lv, rv)


def _ak_rows(merged: ak.Array) -> Counter:
    """Multiset of (k, lv, rv), ``None`` for a null cell — read via the public ``to_list`` surface."""
    out: Counter = Counter()
    for r in ak.to_list(merged):
        out[_triple(r.get("k"), r.get("lv"), r.get("rv"))] += 1
    return out


def _np_rows(merged: object) -> Counter:
    arr = np.ma.asanyarray(merged)
    data, mask = np.ma.getdata(arr), np.ma.getmaskarray(arr)
    names = data.dtype.names or ()

    def cell(name: str, i: int) -> object:
        return None if (name in names and mask[name][i]) else (int(data[name][i]) if name in names else None)

    return Counter(_triple(cell("k", i), cell("lv", i), cell("rv", i)) for i in range(len(data)))


def _pandas_oracle(
    left_k: list[object], left_lv: list[int], right_k: list[object], right_rv: list[int], how: str
) -> Counter:
    """SQL semantics via ``pandas.merge`` — the independent oracle (not the awkward code under test)."""
    dl = pd.DataFrame({"k": pd.array(left_k, dtype="Int64"), "lv": pd.array(left_lv, dtype="Int64")})
    dr = pd.DataFrame({"k": pd.array(right_k, dtype="Int64"), "rv": pd.array(right_rv, dtype="Int64")})
    m = pd.merge(dl, dr, on="k", how=how)

    def cell(col: str, i: int) -> object:
        v = m[col].iloc[i]
        return None if pd.isna(v) else int(v)

    return Counter(_triple(cell("k", i), cell("lv", i), cell("rv", i)) for i in range(len(m)))


def _ak_merge(left: ak.Array, right: ak.Array, how: str) -> ak.Array:
    be = AwkwardBackend()
    bi, pi = be.match_indices(left, right, on=["k"], how=how)
    return be.merge_records(be.take(left, bi), be.take(right, pi), on=["k"])


def _np_merge(left: object, right: object, how: str) -> object:
    be = NumpyBackend()
    bi, pi = be.match_indices(left, right, on=["k"], how=how)
    return be.merge_records(be.take(left, bi), be.take(right, pi), on=["k"])


@pytest.mark.parametrize("name,how,lk,lv,rk,rv,mode", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_awkward_join_option_null_key_matches_sql_oracle(
    name: str, how: str, lk: list[object], lv: list[int], rk: list[object], rv: list[int], mode: str
) -> None:
    """The awkward join primitive on an option ``on`` key with a genuine null must equal the pandas/SQL
    oracle: every present-side row kept, the null key preserved (never dropped/corrupted). Fails on
    HEAD by crash / total row loss / key corruption (``mode``)."""
    oracle = _pandas_oracle(lk, lv, rk, rv, how)
    # The `crash` scenarios RAISE inside merge_records on HEAD — the failure IS the ValueError below.
    merged = _ak_merge(_ak_block(lk, "lv", lv), _ak_block(rk, "rv", rv), how)
    got = _ak_rows(merged)
    assert got == oracle, (
        f"{name} how={how} ({mode}): awkward join diverges from the SQL oracle.\n"
        f"  awkward : {dict(got)}\n  oracle  : {dict(oracle)}"
    )


@pytest.mark.parametrize("name,how,lk,lv,rk,rv,mode", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_awkward_coalesced_key_stays_option_when_a_null_survives(
    name: str, how: str, lk: list[object], lv: list[int], rk: list[object], rv: list[int], mode: str
) -> None:
    """A genuine null key survives into the result, so the coalesced key column MUST stay OPTION-typed
    (its type must carry ``?``). On HEAD ``drop_none`` strips the option (materializes plain ``int64``)
    while corrupting the value — the wrong TYPE. (Crash scenarios raise before this assertion, which is
    itself the HEAD failure.)"""
    merged = _ak_merge(_ak_block(lk, "lv", lv), _ak_block(rk, "rv", rv), how)
    key_type = str(merged["k"].type)
    assert "?" in key_type, (
        f"{name} how={how}: a surviving null key must keep the coalesced key OPTION-typed; got {key_type!r} "
        f"(a plain int64 key cannot represent the null that the SQL oracle keeps)"
    )


@pytest.mark.parametrize("name,how,lk,lv,rk,rv,mode", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_awkward_and_numpy_agree_on_option_null_key_join(
    name: str, how: str, lk: list[object], lv: list[int], rk: list[object], rv: list[int], mode: str
) -> None:
    """M40 two-backend parity: the SAME option-null-key join yields the SAME multiset on the awkward and
    numpy backends, and both equal the pandas/SQL oracle. On HEAD awkward crashes/diverges while numpy
    is correct — so the backends disagree."""
    oracle = _pandas_oracle(lk, lv, rk, rv, how)
    np_got = _np_rows(_np_merge(_np_block(lk, "lv", lv), _np_block(rk, "rv", rv), how))
    assert np_got == oracle, f"{name}: numpy oracle guard — numpy must equal pandas (got {dict(np_got)})"
    ak_got = _ak_rows(_ak_merge(_ak_block(lk, "lv", lv), _ak_block(rk, "rv", rv), how))
    assert ak_got == np_got, (
        f"{name} how={how}: awkward and numpy DIVERGE on an option-null-key join (parity broken).\n"
        f"  awkward: {dict(ak_got)}\n  numpy  : {dict(np_got)}"
    )
