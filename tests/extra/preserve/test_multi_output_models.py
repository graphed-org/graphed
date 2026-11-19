"""A model with several outputs gives its first, one value per event, in every plugin (#244)."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve import JAX_PLUGIN, ONNX_PLUGIN, PYTORCH_PLUGIN, TENSORFLOW_PLUGIN
from graphed.preserve.externals import record_external

X: Any = np.arange(10, dtype="float32").reshape(5, 2)
W1 = np.array([[1.0], [0.0]], dtype="float32")
W2 = np.array([[0.0], [1.0]], dtype="float32")


def _jax() -> tuple[Any, bytes, Any]:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2"), jax.numpy.float32)
    exported = export.export(jax.jit(lambda x: (x @ W1, x @ W2)))(spec)
    return JAX_PLUGIN, bytes(exported.serialize()), None


def _onnx() -> tuple[Any, bytes, Any]:
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", f"w{i}"], [f"y{i}"]) for i in (1, 2)],
        "two",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 2])],
        [helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, ["n", 1]) for i in (1, 2)],
        [numpy_helper.from_array(W1, "w1"), numpy_helper.from_array(W2, "w2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return ONNX_PLUGIN, model.SerializeToString(), {"args": [["$0", "$1"]]}


def _pytorch() -> tuple[Any, bytes, Any]:
    torch = pytest.importorskip("torch")
    w1, w2 = torch.from_numpy(W1), torch.from_numpy(W2)

    two = type("Two", (torch.nn.Module,), {"forward": lambda self, x: (x @ w1, x @ w2)})()
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(two, torch.zeros(1, 2)), buf)
    return PYTORCH_PLUGIN, buf.getvalue(), None


def _tensorflow() -> tuple[Any, bytes, Any]:
    tf = pytest.importorskip("tensorflow")
    x = tf.keras.Input((2,))
    heads = [tf.keras.layers.Dense(1, use_bias=False, name=f"y{i}") for i in (1, 2)]
    model = tf.keras.Model(x, [head(x) for head in heads])
    for head, w in zip(heads, (W1, W2), strict=True):
        head.set_weights([w])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "two.keras"
        model.save(path)
        return TENSORFLOW_PLUGIN, path.read_bytes(), None


@pytest.mark.parametrize("build", [_jax, _onnx, _pytorch, _tensorflow])
def test_a_multi_output_model_gives_its_first_output_per_event(build: Any) -> None:
    plugin, payload, params = build()
    session = Session(AwkwardBackend())
    events = from_awkward("ev", ak.Array({"a": X[:, 0], "b": X[:, 1]}), session=session)
    out = record_external(session, plugin, payload, [events.a, events.b], params=params)
    assert ak.to_list(session.materialize(out)) == (X @ W1).reshape(-1).tolist()
