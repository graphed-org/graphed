"""M27 — variadic call templates against the real ML frameworks (``importorskip`` tier).

Each shipped evaluator must route ``params["args"]`` to a genuinely MULTI-ARGUMENT model —
multiple tensors/avals/feeds, not multiple columns of one matrix — and match the framework's
own prediction exactly. The no-``args`` default stays the m26 single-stacked-matrix convention.
"""

from __future__ import annotations

import io
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed_preserve import JAX_PLUGIN, ONNX_PLUGIN, PYTORCH_PLUGIN, TENSORFLOW_PLUGIN

X0 = np.array([1.0, 2.0, -1.0, 4.0], dtype="float64")
X1 = np.array([0.5, 0.0, 2.0, 1.0], dtype="float64")
MASK = np.array([1.0, 0.0, 1.0, 1.0], dtype="float64")
ARGS2 = {"args": [["$0", "$1"], ["$2"]]}  # arg0 = (n,2) matrix of inputs 0+1; arg1 = (n,1) of input 2


def _eval(plugin: Any, payload: bytes, params: dict[str, Any], inputs: list[Any]) -> np.ndarray:
    resource = plugin.load(payload, params)
    try:
        out = plugin.evaluate(resource, params, inputs)
    finally:
        plugin.close(resource)
    return np.asarray(ak.to_numpy(ak.Array(out)), dtype="float64")


def test_pytorch_two_argument_module() -> None:
    torch = pytest.importorskip("torch")

    class TwoArg(torch.nn.Module):
        def forward(self, kin: Any, mask: Any) -> Any:
            return (kin * 2.0).sum(dim=1) * mask[:, 0]

    traced = torch.jit.trace(TwoArg().eval(), (torch.zeros(1, 2), torch.zeros(1, 1)))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)

    got = _eval(PYTORCH_PLUGIN, buf.getvalue(), ARGS2, [X0, X1, MASK])
    kin = torch.from_numpy(np.stack([X0, X1], axis=1).astype("float32"))
    mask = torch.from_numpy(MASK.astype("float32").reshape(-1, 1))
    with torch.no_grad():
        want = TwoArg()(kin, mask).numpy().reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


def test_pytorch_default_remains_single_stacked_matrix() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.5]]))
        model.bias.fill_(0.0)
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(model.eval(), torch.zeros(1, 2)), buf)
    got = _eval(PYTORCH_PLUGIN, buf.getvalue(), {}, [X0, X1])  # no args: m26 convention
    with torch.no_grad():
        want = model(torch.from_numpy(np.stack([X0, X1], axis=1).astype("float32"))).numpy().reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


def test_tensorflow_two_input_functional_model() -> None:
    keras = pytest.importorskip("keras")
    import pathlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    in_kin = keras.layers.Input(shape=(2,), name="kin")
    in_mask = keras.layers.Input(shape=(1,), name="mask")
    dense = keras.layers.Dense(1, use_bias=False)
    out = keras.layers.Multiply()([dense(in_kin), in_mask])
    model = keras.Model([in_kin, in_mask], out)
    dense.set_weights([np.array([[0.5], [0.25]], dtype="float32")])
    path = pathlib.Path(tempfile.mkdtemp()) / "m.keras"
    model.save(path)

    got = _eval(TENSORFLOW_PLUGIN, path.read_bytes(), ARGS2, [X0, X1, MASK])
    want = np.asarray(
        model([np.stack([X0, X1], axis=1).astype("float32"), MASK.astype("float32").reshape(-1, 1)])
    ).reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


def test_jax_two_aval_export() -> None:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    def f(kin: Any, mask: Any) -> Any:
        return (kin * 2.0).sum(axis=1) * mask[:, 0]

    scope = export.SymbolicScope()
    spec_kin = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2", scope=scope), jax.numpy.float32)
    spec_mask = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1", scope=scope), jax.numpy.float32)
    payload = bytes(export.export(jax.jit(f))(spec_kin, spec_mask).serialize())

    got = _eval(JAX_PLUGIN, payload, ARGS2, [X0, X1, MASK])
    want = np.asarray(f(np.stack([X0, X1], axis=1).astype("float32"), MASK.astype("float32").reshape(-1, 1)))
    assert np.array_equal(got, want.astype("float64"))


def test_onnx_multiple_named_feeds() -> None:
    # onnx + onnxruntime are BASE deps; this still lives in the framework tier for symmetry
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    w = numpy_helper.from_array(np.array([[0.5], [0.25]], dtype=np.float32), name="W")
    kin = helper.make_tensor_value_info("kin", TensorProto.FLOAT, [None, 2])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [None, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["kin", "W"], ["z"]), helper.make_node("Mul", ["z", "mask"], ["y"])],
        "m",
        [kin, mask],
        [y],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    payload = model.SerializeToString()

    params = {"args": {"kin": ["$0", "$1"], "mask": ["$2"]}}
    got = _eval(ONNX_PLUGIN, payload, params, [X0, X1, MASK])
    want = (
        np.stack([X0, X1], axis=1).astype("float32") @ np.array([[0.5], [0.25]], dtype="float32")
    ).reshape(-1) * MASK.astype("float32")
    assert np.allclose(got, want.astype("float64"), atol=1e-7)


def test_onnx_legacy_default_is_unchanged() -> None:
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    w = numpy_helper.from_array(np.array([[2.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="B")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "W", "B"], ["y"])], "m", [x], [y], initializer=[w, b]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    out = _eval(ONNX_PLUGIN, model.SerializeToString(), {}, [X0])  # no args: (-1, 1) single feed
    assert np.array_equal(out, (X0 * 2.0))


# ---------------------------------- keyword arguments ----------------------------------------------
def test_pytorch_keyword_arguments() -> None:
    torch = pytest.importorskip("torch")

    class KwModel(torch.nn.Module):
        def forward(self, kin: Any, mask: Any, scale: Any) -> Any:
            return (kin * 2.0).sum(dim=1) * mask[:, 0] * scale[:, 0]

    traced = torch.jit.trace(KwModel().eval(), (torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 1)))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)

    scale = np.array([2.0, 2.0, 0.5, 1.0], dtype="float64")
    # one positional + two KEYWORD arguments, slots in non-positional order
    params = {"args": [["$0", "$1"]], "kwargs": {"scale": ["$3"], "mask": ["$2"]}}
    got = _eval(PYTORCH_PLUGIN, buf.getvalue(), params, [X0, X1, MASK, scale])
    kin = torch.from_numpy(np.stack([X0, X1], axis=1).astype("float32"))
    with torch.no_grad():
        want = (
            KwModel()(
                kin,
                mask=torch.from_numpy(MASK.astype("float32").reshape(-1, 1)),
                scale=torch.from_numpy(scale.astype("float32").reshape(-1, 1)),
            )
            .numpy()
            .reshape(-1)
        )
    assert np.array_equal(got, want.astype("float64"))


def test_jax_keyword_arguments() -> None:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    def f(kin: Any, *, mask: Any) -> Any:
        return (kin * 2.0).sum(axis=1) * mask[:, 0]

    scope = export.SymbolicScope()
    spec_kin = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2", scope=scope), jax.numpy.float32)
    spec_mask = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1", scope=scope), jax.numpy.float32)
    payload = bytes(export.export(jax.jit(f))(spec_kin, mask=spec_mask).serialize())

    params = {"args": [["$0", "$1"]], "kwargs": {"mask": ["$2"]}}
    got = _eval(JAX_PLUGIN, payload, params, [X0, X1, MASK])
    want = np.asarray(
        f(np.stack([X0, X1], axis=1).astype("float32"), mask=MASK.astype("float32").reshape(-1, 1))
    )
    assert np.array_equal(got, want.astype("float64"))
