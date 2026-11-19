"""A booster that stores feature names and types evaluates as shipped (graphed#232)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed.preserve import XGBOOST_PLUGIN, PreserveError

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


@pytest.mark.parametrize("columns", [{}, {"feature_names": ["a", "b"]}])
@pytest.mark.parametrize("n_inputs", [1, 3])
def test_a_feature_count_mismatch_is_refused(columns: dict[str, Any], n_inputs: int) -> None:
    xgb = pytest.importorskip("xgboost")
    x = np.random.default_rng(1).uniform(0, 12, (300, 2))
    booster = xgb.train(
        {"max_depth": 2, "nthread": 1},
        xgb.DMatrix(x, label=(x.sum(axis=1) > 12).astype("float32"), **columns),
        num_boost_round=2,
    )
    resource = XGBOOST_PLUGIN.load(bytes(booster.save_raw("json")), {})
    try:
        with pytest.raises(PreserveError, match=f"takes 2 features; the call gives {n_inputs}"):
            XGBOOST_PLUGIN.evaluate(resource, {}, [X0] * n_inputs)
    finally:
        XGBOOST_PLUGIN.close(resource)
