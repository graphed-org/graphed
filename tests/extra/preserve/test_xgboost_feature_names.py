"""A booster that stores feature names and types evaluates as shipped (graphed#232)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed.preserve import XGBOOST_PLUGIN

X0 = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
X1 = np.array([1.0, 2.0, 0.0, 2.0, 1.0])


@pytest.mark.parametrize(
    "columns",
    [{"feature_names": ["a", "b"]}, {"feature_names": ["a", "b"], "feature_types": ["q", "c"]}],
)
def test_a_named_booster_predicts_what_it_predicts_eagerly(columns: dict[str, Any]) -> None:
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(3)
    x = np.stack([rng.uniform(0, 12, 300), rng.integers(0, 3, 300)], axis=1)
    y = ((x[:, 0] > 6) & (x[:, 1] > 0)).astype("float32")
    booster = xgb.train(
        {"max_depth": 3, "seed": 0, "nthread": 1},
        xgb.DMatrix(x, label=y, enable_categorical=True, **columns),
        num_boost_round=4,
    )
    payload = bytes(booster.save_raw("json"))
    matrix = np.stack([X0, X1], axis=1)
    want = booster.predict(xgb.DMatrix(matrix, enable_categorical=True, **columns))

    resource = XGBOOST_PLUGIN.load(payload, {})
    try:
        got = XGBOOST_PLUGIN.evaluate(resource, {"args": [["$0", "$1"]]}, [X0, X1])
    finally:
        XGBOOST_PLUGIN.close(resource)
    assert np.array_equal(ak.to_numpy(ak.Array(got)), want.astype("float64"))
