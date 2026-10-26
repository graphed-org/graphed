"""ML External plugins on HiggsDNA's shapes.

A Varied input fans out per universe (GW13), an (n, k) output keeps its rows (GW14), inputs keep
their dtype (GW15), and a gzipped JSON payload hashes and loads as its plain twin (GW16).
"""

from __future__ import annotations

import gzip
import io
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, payloads
from graphed.preserve.externals import (
    CORRECTIONLIB_PLUGIN,
    JAX_PLUGIN,
    ONNX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    TRITON_PLUGIN,
    XGBOOST_PLUGIN,
    ExternalPlugin,
    record_external,
)

X = np.random.default_rng(7).normal(size=(6, 2))
W = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float32)
ARGS = {"args": [["$0", "$1"]]}


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "ev", ak.zip({"a": X[:, 0], "b": X[:, 1]}))


def _onnx(weights: np.ndarray, elem: int = TensorProto.FLOAT) -> bytes:
    k = weights.shape[1]
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "m",
        [helper.make_tensor_value_info("x", elem, [None, 2])],
        [helper.make_tensor_value_info("y", elem, [None, k])],
        initializer=[numpy_helper.from_array(weights, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    return model.SerializeToString()  # type: ignore[no-any-return]


def _onnx_eager(payload: bytes, x: np.ndarray) -> Any:
    sess = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x})[0]


# ---- each plugin with a three-output model, and that model called eagerly -----------------------
def _onnx_case() -> tuple[ExternalPlugin, bytes, dict[str, Any], Callable[[np.ndarray], Any]]:
    payload = _onnx(W)
    return ONNX_PLUGIN, payload, ARGS, lambda x: _onnx_eager(payload, x.astype(np.float32))


def _torch_case() -> tuple[ExternalPlugin, bytes, dict[str, Any], Callable[[np.ndarray], Any]]:
    torch = pytest.importorskip("torch")
    m = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(W.T.copy()))
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(m.eval(), torch.zeros(1, 2)), buf)
    return (
        PYTORCH_PLUGIN,
        buf.getvalue(),
        ARGS,
        lambda x: m(torch.from_numpy(x.astype(np.float32))).detach().numpy(),
    )


def _tensorflow_case() -> tuple[
    ExternalPlugin, bytes, dict[str, Any], Callable[[np.ndarray], Any]
]:
    keras = pytest.importorskip("keras")
    m = keras.Sequential([keras.layers.Input(shape=(2,)), keras.layers.Dense(3, use_bias=False)])
    m.layers[0].set_weights([W])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.keras"
        m.save(path)
        payload = path.read_bytes()
    return TENSORFLOW_PLUGIN, payload, ARGS, lambda x: m(x).numpy()


def _jax_case() -> tuple[ExternalPlugin, bytes, dict[str, Any], Callable[[np.ndarray], Any]]:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2"), jax.numpy.float32)
    exported = export.export(jax.jit(lambda x: x @ W))(spec)
    return JAX_PLUGIN, bytes(exported.serialize()), ARGS, lambda x: exported.call(x)


def _xgboost_case() -> tuple[ExternalPlugin, bytes, dict[str, Any], Callable[[np.ndarray], Any]]:
    xgb = pytest.importorskip("xgboost")
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 3}, xgb.DMatrix(X, label=np.arange(6) % 3), 2
    )
    return (
        XGBOOST_PLUGIN,
        bytes(booster.save_raw("json")),
        ARGS,
        lambda x: booster.predict(xgb.DMatrix(x)),
    )


class _Result:
    def __init__(self, y: np.ndarray) -> None:
        self.y = y

    def as_numpy(self, name: str) -> np.ndarray:
        return self.y


class _Client:
    def infer(self, model: str, inputs: list[Any], outputs: Any = None) -> _Result:
        return _Result(inputs[0].data @ W)

    def close(self) -> None:
        pass


def transport(params: Any) -> _Client:
    return _Client()


class InferInput:
    def __init__(self, name: str, shape: list[int], dt: str) -> None:
        self.data: Any = None

    def set_data_from_numpy(self, a: Any) -> None:
        self.data = np.asarray(a)


class InferRequestedOutput:
    def __init__(self, name: str) -> None:
        self.name = name


def _triton_case() -> tuple[ExternalPlugin, bytes, dict[str, Any], Callable[[np.ndarray], Any]]:
    params = {
        "model": "scorer",
        "url": "http://fake:8000",
        "transport": f"{__name__}:transport",
        "args": {"x": ["$0", "$1"]},
    }
    return TRITON_PLUGIN, b'{"model": "scorer"}', params, lambda x: x.astype(np.float32) @ W


CASES = {
    "onnx": _onnx_case,
    "xgboost": _xgboost_case,
    "torch": _torch_case,
    "tensorflow": _tensorflow_case,
    "jax": _jax_case,
    "triton": _triton_case,
}


def test_a_gzipped_json_model_hashes_and_evaluates_as_its_plain_twin() -> None:
    assert TRITON_PLUGIN.content_hash(
        gzip.compress(b'{"model": "scorer"}')
    ) == TRITON_PLUGIN.content_hash(b'{"model": "scorer"}')
    plugin, payload, params, _eager = _xgboost_case()
    assert plugin.content_hash(gzip.compress(payload)) == plugin.content_hash(payload)
    session, ev = _events()
    plain = session.materialize(
        record_external(session, plugin, payload, [ev.a, ev.b], params=params)
    )
    zipped = session.materialize(
        record_external(session, plugin, gzip.compress(payload), [ev.a, ev.b], params=params)
    )
    assert ak.array_equal(zipped, plain)


@pytest.mark.parametrize("case", CASES)
def test_a_multi_output_model_keeps_one_row_per_event(case: str) -> None:
    plugin, payload, params, eager = CASES[case]()
    session, ev = _events()
    node = record_external(
        session, plugin, payload, [ev.a, ev.b], params=params, output_type="3 * float64"
    )
    got = ak.Array(session.materialize(node))
    want = ak.from_numpy(np.asarray(eager(X), dtype=np.float64))
    assert str(got.type) == str(want.type) == "6 * 3 * float64"
    assert ak.array_equal(got, want)


def test_a_float64_torch_model_runs_in_float64_bit_equal_to_eager() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    m = (
        torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
        .double()
        .eval()
    )
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(m, torch.zeros(1, 2, dtype=torch.float64)), buf)
    session, ev = _events()
    got = ak.to_numpy(
        session.materialize(
            record_external(session, PYTORCH_PLUGIN, buf.getvalue(), [ev.a, ev.b], params=ARGS)
        )
    )
    assert np.array_equal(got, m(torch.from_numpy(X)).detach().numpy().reshape(-1))


def test_a_float64_onnx_model_is_fed_float64_bit_equal_to_eager() -> None:
    # below float32 resolution: a float32 feed would lose it
    weights = W[:, :1].astype(np.float64) + 1e-9
    payload = _onnx(weights, TensorProto.DOUBLE)
    session, ev = _events()
    got = ak.to_numpy(
        session.materialize(
            record_external(session, ONNX_PLUGIN, payload, [ev.a, ev.b], params=ARGS)
        )
    )
    assert np.array_equal(got, _onnx_eager(payload, X).reshape(-1))


def test_an_unfed_optional_string_input_does_not_block_the_fed_ones() -> None:
    string = helper.make_tensor_type_proto(TensorProto.STRING, [None])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "m",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2]),
            helper.make_value_info("tag", helper.make_optional_type_proto(string)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])],
        initializer=[numpy_helper.from_array(W[:, :1], name="W")],
    )
    payload = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9
    ).SerializeToString()
    session, ev = _events()
    params = {"args": {"x": ["$0", "$1"]}}
    got = ak.to_numpy(
        session.materialize(
            record_external(session, ONNX_PLUGIN, payload, [ev.a, ev.b], params=params)
        )
    )
    assert np.array_equal(got, _onnx_eager(payload, X.astype(np.float32)).reshape(-1))


def test_an_external_on_a_varied_input_records_one_node_per_universe() -> None:
    payload = _onnx(W[:, :1])
    session, ev = _events()
    varied = graphed.vary(ev.a, "sc", up=ev.a * 1.1, down=ev.a * 0.9)
    score = record_external(session, ONNX_PLUGIN, payload, [varied, ev.b], params=ARGS)
    assert graphed.labels(score) == graphed.labels(varied)
    for label in graphed.labels(varied):
        a = ak.to_numpy(session.materialize(graphed.universe(varied, label)))
        want = _onnx_eager(payload, np.stack([a, X[:, 1]], axis=1).astype(np.float32)).reshape(-1)
        assert np.array_equal(
            ak.to_numpy(session.materialize(graphed.universe(score, label))), want
        )


CORRECTION = json.dumps(
    {
        "schema_version": 2,
        "corrections": [
            {
                "name": "sf",
                "version": 1,
                "inputs": [{"name": "x", "type": "real"}],
                "output": {"name": "w", "type": "real"},
                "data": {
                    "nodetype": "formula",
                    "expression": "1.5*x",
                    "parser": "TFormula",
                    "variables": ["x"],
                },
            }
        ],
    }
).encode()


def _unused(*args: object) -> object:  # with args=, the correctionlib plugin evaluates
    raise AssertionError


def test_a_gzipped_correction_set_is_its_plain_twin() -> None:
    gz = gzip.compress(CORRECTION)
    plain, zipped = (payloads.correctionlib_contents_descriptor(p, "sf") for p in (CORRECTION, gz))
    assert (zipped.content_hash, zipped.version) == (plain.content_hash, plain.version)
    assert CORRECTIONLIB_PLUGIN.content_hash(gz) == plain.content_hash
    session, ev = _events()
    got = session.materialize(gak.apply_correction(gz, "sf", [ev.a], _unused, args=["$0"]))
    assert ak.to_list(got) == pytest.approx(list(1.5 * X[:, 0]))
