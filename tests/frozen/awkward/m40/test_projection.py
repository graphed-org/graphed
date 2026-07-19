"""M40 — (d) column projection through a JOIN (plan §3.4; extends the M5 over-touch guard).

A join must read EXACTLY ``{key | used}`` on EACH side and never more — the two-source form of the
dask-awkward over-touch bug M5 exists to prevent. Here the key is ``(run, lumi, event)`` (packed per
side) and the downstream expression uses only ``lv`` (left) and ``rv`` (right). The unused columns
``lx``/``rx`` must be absent from both read sets.

Discrimination: an impl whose Join arm gathers the FULL merged record on the reporting typetracer
(carrying ``lx``/``rx`` through) over-touches -> ``lx``/``rx`` appear in the read set -> FAILS. An
impl that never reads the key columns to route the join under-touches -> the ``run/lumi/event``
assertion FAILS.
"""

from __future__ import annotations

import awkward as ak

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.awkward.projection import project


def _join_out() -> object:
    s = Session(AwkwardBackend())
    left = from_awkward(
        s,
        "left",
        ak.Array({"run": [1, 1], "lumi": [1, 1], "event": [1, 2], "lv": [10, 20], "lx": [7, 8]}),
    )
    right = from_awkward(
        s,
        "right",
        ak.Array({"run": [1, 1], "lumi": [1, 1], "event": [1, 2], "rv": [100, 200], "rx": [9, 9]}),
    )
    joined = gak.join(left, right, on=["run", "lumi", "event"], how="inner")
    return joined.lv + joined.rv  # uses lv (left) + rv (right); key read by the join itself


def test_join_reads_exactly_key_plus_used_on_the_left() -> None:
    proj = project(_join_out())
    assert proj.columns_for("left") == frozenset({"run", "lumi", "event", "lv"})
    assert "lx" not in proj.columns_for("left"), "unused left column must not be over-touched"


def test_join_reads_exactly_key_plus_used_on_the_right() -> None:
    proj = project(_join_out())
    assert proj.columns_for("right") == frozenset({"run", "lumi", "event", "rv"})
    assert "rx" not in proj.columns_for("right"), "unused right column must not be over-touched"


def test_neither_side_pulls_the_other_sides_columns() -> None:
    # the seam is real: left's read set carries no rv/rx, right's carries no lv/lx.
    proj = project(_join_out())
    assert not (proj.columns_for("left") & {"rv", "rx"})
    assert not (proj.columns_for("right") & {"lv", "lx"})
