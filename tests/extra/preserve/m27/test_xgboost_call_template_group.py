"""M27 xgboost gap: the valid single-group call template never runs a real booster.

``tests/frozen/preserve/m27/test_variadic_call_templates.py`` only exercises xgboost's call
template through the loud-rejection path (multiple positional args). The one successful shape —
a single positional GROUP that stacks several node inputs into xgboost's one tabular matrix —
was never run against a real booster.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed.preserve import XGBOOST_PLUGIN

X0 = np.array([0.0, 3.0, 6.0, 9.0, 12.0], dtype="float64")
X1 = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype="float64")


def _eval(payload: bytes, params: dict[str, Any], inputs: list[Any]) -> Any:
    resource = XGBOOST_PLUGIN.load(payload, params)
    try:
        out = XGBOOST_PLUGIN.evaluate(resource, params, inputs)
    finally:
        XGBOOST_PLUGIN.close(resource)
    return np.asarray(ak.to_numpy(ak.Array(out)), dtype="float64")


def test_xgboost_single_group_template_stacks_and_matches_booster() -> None:
    xgb = pytest.importorskip("xgboost")

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 12, size=(300, 2))
    y = ((x[:, 0] > 6) & (x[:, 1] > 6)).astype("float32")  # needs BOTH columns to predict
    booster = xgb.train(
        {"max_depth": 3, "objective": "binary:logistic", "seed": 0, "nthread": 1},
        xgb.DMatrix(x, label=y),
        num_boost_round=8,
    )
    payload = bytes(booster.save_raw("json"))
    reloaded = xgb.Booster()
    reloaded.load_model(bytearray(payload))
    assert set(reloaded.get_score(importance_type="weight")) == {"f0", "f1"}, (
        "the fixture model must split on both features for the group-stacking test to discriminate"
    )

    got = _eval(payload, {"args": [["$0", "$1"]]}, [X0, X1])

    want_matrix = np.stack([X0, X1], axis=1).astype("float32")
    want = reloaded.predict(xgb.DMatrix(want_matrix))

    assert np.array_equal(got, want.astype("float64"))
    # the group must actually route through ml_matrix's stacking (both columns, in order), not
    # just re-run the legacy single-column convention or swap the columns
    single_col = reloaded.predict(xgb.DMatrix(X0.astype("float32").reshape(-1, 1)))
    assert not np.array_equal(got, single_col.astype("float64"))
    swapped = _eval(payload, {"args": [["$1", "$0"]]}, [X0, X1])
    assert not np.array_equal(got, swapped)
