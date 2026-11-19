"""A model External on a partition with zero rows gives zero rows of its output type (#241)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import StageError
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


def _jax(k: int, batch: str = "b") -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    weights: Any = np.ones((2, k), dtype="float32")
    spec = jax.ShapeDtypeStruct(export.symbolic_shape(f"{batch}, 2"), jax.numpy.float32)
    return bytes(export.export(jax.jit(lambda x: x @ weights))(spec).serialize())


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(
    ("plugin", "build"),
    [
        (XGBOOST_PLUGIN, _xgboost),
        (JAX_PLUGIN, _jax),
        (JAX_PLUGIN, lambda k: _jax(k, batch="2*b")),
    ],
)
def test_zero_rows_give_zero_rows_of_the_declared_type(plugin: Any, build: Any, k: int) -> None:
    payload = build(k)
    session = Session(AwkwardBackend())
    events = from_awkward("ev", ak.Array({"a": X[:, 0], "b": X[:, 1]}), session=session)
    events = events[events.a > 99]
    out = record_external(session, plugin, payload, [events.a, events.b], output_type=DECLARED[k])
    value: Any = session.materialize(out)
    assert len(value) == 0
    assert str(ak.type(value).content) == DECLARED[k]


def test_a_width_that_depends_on_the_batch_is_not_guessed() -> None:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2"), jax.numpy.float32)
    payload = bytes(export.export(jax.jit(lambda x: x @ x.T))(spec).serialize())
    session = Session(AwkwardBackend())
    events = from_awkward("ev", ak.Array({"a": X[:, 0], "b": X[:, 1]}), session=session)
    events = events[events.a > 99]
    out = record_external(session, JAX_PLUGIN, payload, [events.a, events.b])
    with pytest.raises(StageError, match="dimension variable 'b'"):
        session.materialize(out)
