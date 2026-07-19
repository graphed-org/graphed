"""M40 — (a4) the awkward-only ``gak.join(..., grouped=True)`` convenience (plan §3.1, §3.3).

``grouped=True`` is a pure post-op over the RELATIONAL result: regroup by a deterministic
``ak.unflatten`` whose run-lengths are the per-build-row match counts. So flattening the grouped
result must reproduce the relational result exactly, and the outer length must equal the number of
build rows that matched (not the number of output rows).

Witness / discrimination:
  * ``len(grouped) == build-rows`` and ``ak.num(grouped, axis=1) == match counts`` — an impl that
    returns the ungrouped relational result for ``grouped=True`` has ``len == 4`` (output rows) and
    FAILS both;
  * flattening the grouped result equals the ungrouped relational multiset — pins grouped as
    literally ``unflatten(relational, counts)``, deterministic (no arrival/order dependence).
The grouped shape is ONLY here (awkward post-op); the neutral ``ShuffleBackend`` contract stays
relational (see ``test_join_primitives.py``), so this convenience is correctly ABSENT on numpy.
"""

from __future__ import annotations

import awkward as ak

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak


def _sources(session: Session) -> tuple[object, object]:
    # key = (run, lumi, event); left row0 = key(1,1,1) has NO right match, rows1,2 = key(1,1,2)
    # each match BOTH right rows (also key(1,1,2)) -> inner: 0 + 2x2 = 4 output rows, 2 build rows.
    left = from_awkward(
        session,
        "left",
        ak.Array({"run": [1, 1, 1], "lumi": [1, 1, 1], "event": [1, 2, 2], "lv": [10, 20, 21]}),
    )
    right = from_awkward(
        session, "right", ak.Array({"run": [1, 1], "lumi": [1, 1], "event": [2, 2], "rv": [200, 201]})
    )
    return left, right


def test_grouped_join_is_the_relational_result_regrouped_by_unflatten() -> None:
    s = Session(AwkwardBackend())
    left, right = _sources(s)
    flat = gak.join(left, right, on=["run", "lumi", "event"], how="inner")
    grouped = gak.join(left, right, on=["run", "lumi", "event"], how="inner", grouped=True)

    fr = s.materialize(flat)
    gr = s.materialize(grouped)

    # relational baseline: 4 duplicated rows for the matched key.
    assert len(fr) == 4
    flat_pairs = sorted(zip(fr["lv"].to_list(), fr["rv"].to_list(), strict=True))
    assert flat_pairs == sorted([(20, 200), (20, 201), (21, 200), (21, 201)])

    # grouped: one sublist per matching BUILD row (2), run-lengths = match counts (2, 2) — NOT a
    # flat 4-row array (which is what an ungrouped impl would return for grouped=True).
    assert len(gr) == 2, "grouped by build row: 2 sublists, not 4 flat rows"
    assert sorted(ak.num(gr, axis=1).to_list()) == [2, 2], "run-lengths are the per-build match counts"

    # flattening the grouping reproduces the relational result bit-for-bit (deterministic unflatten).
    gflat = ak.flatten(gr, axis=1)
    assert sorted(zip(gflat["lv"].to_list(), gflat["rv"].to_list(), strict=True)) == flat_pairs
