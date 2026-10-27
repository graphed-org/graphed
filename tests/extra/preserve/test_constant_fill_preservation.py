"""A fill with scalar constants preserves and reproduces.

The plugin rebuilds its hash and evaluator.
"""

from __future__ import annotations

import json

import awkward as ak
import boost_histogram as bh
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve import build_bundle, reproduce
from graphed.preserve.externals.histogram_external import _canonical_payload

gh = pytest.importorskip("graphed_histogram")

needs_constants = pytest.mark.skipif(
    "constants" not in gh.FillEvaluator.__dataclass_fields__,
    reason="this graphed-histogram records no fill constants",
)

EVENTS = ak.Array({"x": [1.0, 4.0, 7.0, 2.5] * 20, "w": [0.5, 1.0, 2.0, 1.5] * 20})
X, W = np.asarray(EVENTS.x), np.asarray(EVENTS.w)


def test_the_canonical_payload_folds_constants_in_last() -> None:
    constants = json.dumps({"axes": [[1, 1.5]]})
    params = {"spec": "S", "weighted": False, "n_weights": 2, "constants": constants}
    assert _canonical_payload(params) == "S\x00unweighted\x00n_weights=2\x00constants=" + constants


@needs_constants
@pytest.mark.parametrize(
    ("axes", "fill", "eager"),
    [
        (
            (bh.axis.Regular(4, 0.0, 8.0), bh.axis.Regular(3, 0.0, 3.0)),
            lambda h, g: h.fill(g.x, 1.5, weight=[g.w, 2.0]),
            lambda h: h.fill(X, 1.5, weight=W * 2.0),
        ),
        (
            (bh.axis.StrCategory([], growth=True), bh.axis.Regular(4, 0.0, 8.0)),
            lambda h, g: h.fill("DYJets", g.x, weight=g.w),
            lambda h: h.fill("DYJets", X, weight=W),
        ),
    ],
    ids=["constant-axis-and-weight", "constant-category"],
)
def test_a_constant_fill_preserves_and_reproduces(tmp_path, axes, fill, eager) -> None:  # type: ignore[no-untyped-def]
    session = Session(AwkwardBackend())
    deferred = gh.boost.Histogram(*axes, storage=bh.storage.Weight())
    fill(deferred, from_awkward(session, "events", EVENTS))
    (node,) = deferred.fill_nodes()
    bundle = build_bundle(
        tmp_path / "b", session=session, value=node, datasets={"events": EVENTS}, payloads={}
    )
    want = bh.Histogram(*axes, storage=bh.storage.Weight())
    eager(want)
    assert reproduce(bundle) == want
