"""M40 theme (a3) — the numpy validity carrier and THE ``np.take(-1)`` trap (spec TRAP #1, E4).

A left/outer join with an unmatched row must read the missing field as **null/invalid**, never a real
value. The trap (ADV-r5.1): ``match_indices`` marks a miss with index ``-1``; if ``take`` were
``np.take`` it would gather the LAST row — a real value — and a row-count-only check would pass. So
every assertion here is on the **validity bit**, not the row count. E4 requires the mask to round-trip
through ``to_wire``/``from_wire`` (Arrow native validity) and to survive ``concat``/``slice_rows``;
dropping it silently corrupts left/outer on recombine/spill (ADV-r6.3).
"""

from __future__ import annotations

import numpy as np

from graphed.numpy import NumpyBackend

_KEY = "__joinkey__"


def _rec(keys: list[int], **cols: object) -> np.ndarray:
    dt = [(_KEY, np.uint64), *[(k, np.asarray(v).dtype) for k, v in cols.items()]]
    a = np.zeros(len(keys), dtype=dt)
    a[_KEY] = np.asarray(keys, dtype=np.uint64)
    for k, v in cols.items():
        a[k] = np.asarray(v)
    return a


def _valid(block: object, field: str) -> np.ndarray:
    """The E4 per-row validity bit for ``field``: True where present, False where null.

    Reads the option carrier's mask. A plain (unmasked) result — what an ``np.take``-based ``take``
    returns — reports all-True here, so any assertion that a miss is invalid FAILS that buggy impl."""
    col = np.ma.asanyarray(block)[field]
    return ~np.ma.getmaskarray(col)


def test_take_maps_negative_one_to_invalid_not_the_last_row() -> None:
    # THE trap. index -1 is a MISS, not "the last row". The last row carries a distinctive sentinel
    # (999); an np.take-based impl would gather it AND leave the row valid — both are asserted against.
    be = NumpyBackend()
    block = _rec([10, 20, 30, 777], v=[0, 1, 2, 999])
    out = be.take(block, np.array([1, -1, 0]))
    valid = _valid(out, "v")
    assert bool(valid[0]) and bool(valid[2])  # real gathers stay valid
    assert not bool(valid[1])  # the -1 row reads INVALID (np.take impl leaves it True → fails)
    assert list(np.ma.getdata(np.ma.asanyarray(out)["v"])[[0, 2]]) == [1, 0]  # correct values gathered
    assert int(np.ma.getdata(np.ma.asanyarray(out)["v"])[1]) != 999  # did NOT gather the last row


def _outer_join(be: NumpyBackend, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    b, p = be.match_indices(left, right, on=[_KEY], how="outer")
    return be.merge_records(be.take(left, b), be.take(right, p), on=[_KEY])


def test_outer_join_missing_field_reads_null_on_each_side() -> None:
    # outer join, disjoint-plus-overlap keys: exactly one row is right-only (its lx is null) and one is
    # left-only (its rx is null); the matched row has neither null. Asserted on VALIDITY, not row count.
    be = NumpyBackend()
    left = _rec([10, 20], lx=np.array([100, 200], dtype=np.int64))
    right = _rec([20, 30], rx=np.array([2.0, 3.0], dtype=np.float64))
    merged = _outer_join(be, left, right)
    lx_valid, rx_valid = _valid(merged, "lx"), _valid(merged, "rx")
    assert int((~lx_valid).sum()) == 1  # the right-only (key 30) row has a null lx
    assert int((~rx_valid).sum()) == 1  # the left-only (key 10) row has a null rx
    assert int((lx_valid & rx_valid).sum()) == 1  # the matched (key 20) row: both present
    # an np.take(-1) impl produces zero nulls here (it fabricates last-row values) → the counts fail.


def test_validity_survives_wire_roundtrip() -> None:
    # E4: to_wire/from_wire carry the option block via Arrow native validity. The .npy path (M39) would
    # silently drop the mask on save — so a lost bit here is the discriminating failure.
    be = NumpyBackend()
    block = _rec([10, 20, 30], v=[0, 1, 2])
    opt = be.take(block, np.array([1, -1, 2]))  # row 1 is null
    back = be.from_wire(be.to_wire(opt))
    v = _valid(back, "v")
    assert list(v) == [True, False, True]  # the null SURVIVED the wire
    assert list(np.ma.getdata(np.ma.asanyarray(back)["v"])[[0, 2]]) == [1, 2]


def test_validity_survives_concat_and_slice_rows() -> None:
    # E4/ADV-r6.3: recombining (concat) and re-slicing option blocks must preserve the mask. numpy's
    # plain np.concatenate / fancy-slice drop it → the pattern below flattens to all-True and fails.
    be = NumpyBackend()
    block = _rec([10, 20, 30], v=[0, 1, 2])
    a = be.take(block, np.array([0, -1]))  # [valid, null]
    b = be.take(block, np.array([-1, 2]))  # [null, valid]
    merged = be.concat([a, b])
    assert list(_valid(merged, "v")) == [True, False, False, True]  # mask preserved across concat
    sl = be.slice_rows(merged, 1, 3)
    assert list(_valid(sl, "v")) == [False, False]  # and across a contiguous slice
