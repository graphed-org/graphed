"""m68c §3.3 (``graphed.awkward.join.take``): a per-dest join over hash-partitioned ``pack_key`` sides
has the whole join's type, also at a dest whose left or right side has no rows, so the union of the
per-dest joins is the whole join."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from m68c_services_harness import rows, sublists

from graphed.awkward import AwkwardBackend
from graphed.shuffle import JOINKEY

PARTS = 8
CASES = [(False, "inner"), (False, "left"), (False, "right"), (False, "outer"), (True, "inner"), (True, "left")]


@pytest.mark.parametrize(("grouped", "how"), CASES)
def test_per_dest_joins_concatenate_to_the_whole_join(grouped: bool, how: str) -> None:
    be = AwkwardBackend()
    # run 7 exists only on the left, runs 5 and 6 only on the right
    left = be.eval_stage("pack_key", [ak.Array({"run": np.array([1, 2, 3, 4, 7]), "w": np.arange(5.0)})], {"on": "run"})
    right = be.eval_stage(
        "pack_key", [ak.Array({"run": np.array([1, 1, 2, 4, 4, 5, 6]), "x": np.arange(7.0)})], {"on": "run"}
    )
    lp, rp = be.partition(left, JOINKEY, PARTS), be.partition(right, JOINKEY, PARTS)
    sizes = [(len(a), len(b)) for a, b in zip(lp, rp, strict=True)]
    assert any(n == 0 < m for n, m in sizes)
    assert any(m == 0 < n for n, m in sizes)

    params = {"on": f"run,{JOINKEY}", "how": how, "grouped": grouped}
    whole = ak.Array(be.eval_stage("join", [left, right], params))
    union = ak.concatenate([be.eval_stage("join", [lp[d], rp[d]], params) for d in range(PARTS)])
    assert str(union.type) == str(whole.type)
    assert (sublists if grouped else rows)(union) == (sublists if grouped else rows)(whole)
