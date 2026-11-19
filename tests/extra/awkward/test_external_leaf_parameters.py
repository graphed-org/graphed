"""An External's recorded type drops an input's leaf parameters, as its value does (#248)."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.preserve import (
    JAX_PLUGIN,
    ONNX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    XGBOOST_PLUGIN,
)
from graphed.preserve.externals import record_external

DOUBLE = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "c",
                "version": 1,
                "inputs": [{"name": "x", "type": "real"}],
                "output": {"name": "w", "type": "real"},
                "data": {
                    "nodetype": "formula",
                    "expression": "2*x",
                    "parser": "TFormula",
                    "variables": ["x"],
                },
            }
        ],
    }
).encode()
JAGGED = ak.Array([[1.0, 2.0], [], [3.0]]).layout


def _types(session: Session, out: Any) -> tuple[str, str]:
    recorded = str(session.form(out).describe()).removeprefix("## * ")
    return recorded, str(ak.type(session.materialize(out)).content)


@pytest.mark.parametrize(
    ("content", "lists", "expected"),
    [
        ({"typename": "float[]"}, None, "var * float64"),
        (None, {"__doc__": "jets"}, '[var * float64, parameters={"__doc__": "jets"}]'),
    ],
    ids=["leaf", "list"],
)
def test_a_correction_drops_leaf_parameters_and_keeps_list_ones(
    content: Any, lists: Any, expected: str
) -> None:
    leaf = ak.contents.NumpyArray(JAGGED.content.data, parameters=content)
    array = ak.Array(ak.contents.ListOffsetArray(JAGGED.offsets, leaf, parameters=lists))
    session = Session(AwkwardBackend())
    out = gak.apply_correction(DOUBLE, "c", [from_awkward("x", array, session=session)])
    assert _types(session, out) == (expected, expected)


def _onnx() -> bytes:
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 1])],
        [numpy_helper.from_array(np.array([2.0], dtype="float32"), "two")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return bytes(model.SerializeToString())


def _xgboost() -> bytes:
    xgb = pytest.importorskip("xgboost")
    data = xgb.DMatrix(np.arange(8.0).reshape(-1, 1), label=np.arange(8.0))
    return bytes(xgb.train({"max_depth": 1}, data, 1).save_raw("json"))


def _jax() -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1"), jax.numpy.float32)
    return bytes(export.export(jax.jit(lambda x: x * 2))(spec).serialize())


def _pytorch() -> bytes:
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(torch.nn.Linear(1, 1), torch.zeros(1, 1)), buf)
    return buf.getvalue()


def _tensorflow() -> bytes:
    tf = pytest.importorskip("tensorflow")
    x = tf.keras.Input((1,))
    model = tf.keras.Model(x, tf.keras.layers.Dense(1)(x))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.keras"
        model.save(path)
        return path.read_bytes()


@pytest.mark.parametrize(
    ("plugin", "build"),
    [
        (ONNX_PLUGIN, _onnx),
        (XGBOOST_PLUGIN, _xgboost),
        (JAX_PLUGIN, _jax),
        (PYTORCH_PLUGIN, _pytorch),
        (TENSORFLOW_PLUGIN, _tensorflow),
    ],
    ids=["onnx", "xgboost", "jax", "pytorch", "tensorflow"],
)
def test_a_model_output_drops_its_inputs_leaf_parameters(plugin: Any, build: Any) -> None:
    payload = build()
    session = Session(AwkwardBackend())
    x = from_awkward(
        "x", ak.with_parameter(ak.Array([1.0, 2.0, 3.0]), "typename", "float"), session=session
    )
    assert _types(session, record_external(session, plugin, payload, [x])) == (
        "float64",
        "float64",
    )
