"""M40 theme (d) — column projection through a join, BOTH sides (§B-15, M5 over-touch guard).

A join reads exactly ``{key | used}`` on each side: the join-key columns (consumed by ``pack_key``)
plus whatever downstream columns are actually referenced. Anything else is over-touch — the M5 defect
this guards against, now doubled because a join has two independent record sources. The unused columns
below (``lx_unused`` / ``rx_unused``) must never appear in either source's projected column set.
"""

from __future__ import annotations

import numpy as np

import graphed
from graphed import Session
from graphed.numpy import NumpyBackend, from_record, project


def _sources() -> tuple[object, object]:
    s = Session(NumpyBackend())
    left = from_record(
        s,
        "left",
        run=np.array([1, 1, 2], dtype=np.int64),
        lumi=np.array([10, 10, 20], dtype=np.int64),
        event=np.array([100, 101, 102], dtype=np.int64),
        lx_used=np.array([1.0, 2.0, 3.0]),
        lx_unused=np.array([9.0, 9.0, 9.0]),
    )
    right = from_record(
        s,
        "right",
        run=np.array([1, 1, 2], dtype=np.int64),
        lumi=np.array([10, 10, 20], dtype=np.int64),
        event=np.array([100, 101, 102], dtype=np.int64),
        rx_used=np.array([4.0, 5.0, 6.0]),
        rx_unused=np.array([7.0, 7.0, 7.0]),
    )
    return left, right


def test_join_projects_exactly_key_union_used_on_both_sides() -> None:
    left, right = _sources()
    joined = graphed.join(left, right, on=["run", "lumi", "event"], how="inner")
    expr = joined["lx_used"] + joined["rx_used"]  # uses one non-key column from each side
    proj = project(expr)
    left_cols = proj.columns_for("left")
    right_cols = proj.columns_for("right")
    assert left_cols == frozenset({"run", "lumi", "event", "lx_used"})
    assert right_cols == frozenset({"run", "lumi", "event", "rx_used"})
    # explicit over-touch guard: the untouched columns are read on NEITHER side.
    assert "lx_unused" not in left_cols
    assert "rx_unused" not in right_cols
