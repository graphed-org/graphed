"""A model External on a partition with zero rows gives zero rows of its output type (#241)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve import JAX_PLUGIN, XGBOOST_PLUGIN
from graphed.preserve.externals import record_external

X = np.random.default_rng(0).normal(size=(64, 2)).astype("float32")
DECLARED = {1: "float64", 3: "3 * float64"}


def _xgboost(k: int) -> bytes:
    xgb = pytest.importorskip("xgboost")
    multi = {"objective": "multi:softprob", "num_class": 3}
    params = {"max_depth": 2, **(multi if k == 3 else {"objective": "binary:logistic"})}
    labels: Any = np.arange(64) % (3 if k == 3 else 2)
    return bytes(xgb.train(params, xgb.DMatrix(X, label=labels), 2).save_raw("json"))


def _jax(k: int) -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    weights: Any = np.ones((2, k), dtype="float32")
    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2"), jax.numpy.float32)
    return bytes(export.export(jax.jit(lambda x: x @ weights))(spec).serialize())


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(("plugin", "build"), [(XGBOOST_PLUGIN, _xgboost), (JAX_PLUGIN, _jax)])
def test_zero_rows_give_zero_rows_of_the_declared_type(plugin: Any, build: Any, k: int) -> None:
    payload = build(k)
    session = Session(AwkwardBackend())
    events = from_awkward("ev", ak.Array({"a": X[:, 0], "b": X[:, 1]}), session=session)
    events = events[events.a > 99]
    out = record_external(session, plugin, payload, [events.a, events.b], output_type=DECLARED[k])
    value: Any = session.materialize(out)
    assert len(value) == 0
    assert str(ak.type(value).content) == DECLARED[k]


def test_a_batch_framework_that_refuses_zero_rows_still_answers_them() -> None:
    from graphed.preserve.externals._helpers import predict_rows  # noqa: PLC0415

    def predict(x: Any, *, y: Any) -> Any:
        if len(x) == 0:
            raise ValueError("empty batch")
        return np.ones((len(x), 3)) * x[:, :1] * y[:, :1]

    full = predict_rows(predict, np.ones((2, 2)), y=np.ones((2, 1)))
    assert full.shape == (2, 3)
    empty = predict_rows(predict, np.ones((0, 2)), y=np.ones((0, 1)))
    assert empty.shape == (0, 3)
