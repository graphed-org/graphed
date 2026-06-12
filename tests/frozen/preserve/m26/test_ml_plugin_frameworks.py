"""M26 — the framework tier: each shipped plugin against its real framework.

Per framework (``pytest.importorskip``; install via ``pip install -e .[mltest]``): the hash is
**content identity, not byte identity** — stable across re-saves of the same model (archive
bytes are allowed to differ; TorchScript/.keras zips and jax.export blobs all embed volatile
metadata), sensitive to the weights AND to the architecture — and the shipped evaluator
reproduces the framework's own prediction exactly. ``validate_plugin`` runs the full
cross-process determinism + non-vacuity check on every one.
"""

from __future__ import annotations

import io
import pathlib
import tempfile
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed_preserve import (
    JAX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    XGBOOST_PLUGIN,
    validate_plugin,
)

FEATS = np.array([0.0, 1.0, 2.5, -1.0, 4.0], dtype="float64")


def _evaluated(
    plugin: Any, payload: bytes, inputs: list[Any], params: dict[str, Any] | None = None
) -> np.ndarray:
    resource = plugin.load(payload, params or {})
    try:
        out = plugin.evaluate(resource, params or {}, inputs)
    finally:
        plugin.close(resource)
    return np.asarray(ak.to_numpy(ak.Array(out)), dtype="float64")


# ---------------------------------- PyTorch (TorchScript) ----------------------------------------
def _torch_bytes(weight: float, *, extra_layer: bool = False) -> bytes:
    import torch  # noqa: PLC0415

    layers: list[Any] = [torch.nn.Linear(1, 1), torch.nn.Sigmoid()]
    if extra_layer:
        layers.append(torch.nn.Sigmoid())  # same parameters, different architecture
    model = torch.nn.Sequential(*layers)
    with torch.no_grad():
        model[0].weight.fill_(weight)
        model[0].bias.fill_(0.1)
    model.eval()
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(model, torch.zeros(1, 1)), buf)
    return buf.getvalue()


def test_pytorch_hash_is_content_identity() -> None:
    pytest.importorskip("torch")
    h = PYTORCH_PLUGIN.content_hash
    a, a_resaved, b_weights, c_arch = (
        _torch_bytes(0.5),
        _torch_bytes(0.5),
        _torch_bytes(0.9),
        _torch_bytes(0.5, extra_layer=True),
    )
    assert a != a_resaved, "TorchScript archives are volatile — the premise of content hashing"
    assert h(a) == h(a_resaved)  # same model, different bytes -> SAME identity
    assert h(a) != h(b_weights)  # weights are content
    assert h(a) != h(c_arch)  # architecture is content (same state_dict!)
    validate_plugin(PYTORCH_PLUGIN)


def test_pytorch_evaluator_matches_torch_exactly() -> None:
    torch = pytest.importorskip("torch")
    payload = _torch_bytes(0.4)
    got = _evaluated(PYTORCH_PLUGIN, payload, [FEATS])
    model = torch.jit.load(io.BytesIO(payload))
    with torch.no_grad():
        want = model(torch.from_numpy(FEATS.astype("float32").reshape(-1, 1))).numpy().reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


def test_pytorch_evaluator_stacks_multiple_inputs() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.5]]))
        model.bias.fill_(0.0)
    model.eval()
    buf = io.BytesIO()
    torch.jit.save(torch.jit.trace(model, torch.zeros(1, 2)), buf)
    x0, x1 = FEATS, FEATS * 2.0 + 1.0
    got = _evaluated(PYTORCH_PLUGIN, buf.getvalue(), [x0, x1])
    with torch.no_grad():
        cols = np.stack([x0.astype("float32"), x1.astype("float32")], axis=1)
        want = model(torch.from_numpy(cols)).numpy().reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


# ---------------------------------- TensorFlow (.keras) ------------------------------------------
def _keras_bytes(weight: float, *, extra_layer: bool = False) -> bytes:
    keras = pytest.importorskip("keras")

    layers = [keras.layers.Input(shape=(1,)), keras.layers.Dense(1, activation="sigmoid")]
    if extra_layer:
        layers.append(keras.layers.Activation("sigmoid"))  # parameter-free architecture change
    model = keras.Sequential(layers)
    model.layers[0].set_weights([np.array([[weight]], dtype="float32"), np.array([0.1], dtype="float32")])
    path = pathlib.Path(tempfile.mkdtemp()) / "model.keras"
    model.save(path)
    return path.read_bytes()


def test_tensorflow_hash_is_content_identity() -> None:
    pytest.importorskip("keras")
    h = TENSORFLOW_PLUGIN.content_hash
    a, a_resaved, b_weights, c_arch = (
        _keras_bytes(0.5),
        _keras_bytes(0.5),
        _keras_bytes(0.9),
        _keras_bytes(0.5, extra_layer=True),
    )
    assert a != a_resaved, ".keras archives are volatile — the premise of content hashing"
    assert h(a) == h(a_resaved)  # auto-generated layer NAMES are formatting, not content
    assert h(a) != h(b_weights)
    assert h(a) != h(c_arch)
    validate_plugin(TENSORFLOW_PLUGIN)


def test_tensorflow_evaluator_matches_keras_exactly() -> None:
    keras = pytest.importorskip("keras")
    payload = _keras_bytes(0.4)
    got = _evaluated(TENSORFLOW_PLUGIN, payload, [FEATS])
    path = pathlib.Path(tempfile.mkdtemp()) / "model.keras"
    path.write_bytes(payload)
    model = keras.models.load_model(path, compile=False)
    want = np.asarray(model(FEATS.astype("float32").reshape(-1, 1))).reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


# ---------------------------------- XGBoost (JSON model) -----------------------------------------
def _xgb_bytes(seed: int) -> bytes:
    xgb = pytest.importorskip("xgboost")

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 12, size=(300, 1))
    y = (x[:, 0] > 6).astype("float32")
    booster = xgb.train(
        {"max_depth": 2, "objective": "binary:logistic", "seed": 0, "nthread": 1},
        xgb.DMatrix(x, label=y),
        num_boost_round=4,
    )
    return bytes(booster.save_raw("json"))


def test_xgboost_hash_is_stable_and_model_sensitive() -> None:
    h = XGBOOST_PLUGIN.content_hash
    a, a_again, b = _xgb_bytes(1), _xgb_bytes(1), _xgb_bytes(2)
    assert h(a) == h(a_again)
    assert h(a) != h(b)


def test_xgboost_evaluator_matches_booster_exactly() -> None:
    xgb = pytest.importorskip("xgboost")
    payload = _xgb_bytes(3)
    got = _evaluated(XGBOOST_PLUGIN, payload, [FEATS])
    booster = xgb.Booster()
    booster.load_model(bytearray(payload))
    want = booster.predict(xgb.DMatrix(FEATS.astype("float32").reshape(-1, 1)))
    assert np.array_equal(got, want.astype("float64"))


# ---------------------------------- JAX (jax.export) ---------------------------------------------
def _jax_bytes(weight: float, *, extra_op: bool = False) -> bytes:
    jax = pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    def f(x: Any) -> Any:
        y = jax.nn.sigmoid(weight * x[:, 0] + 0.1)
        return y * 2.0 if extra_op else y

    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1"), jax.numpy.float32)
    return bytes(export.export(jax.jit(f))(spec).serialize())


def test_jax_hash_is_content_identity() -> None:
    pytest.importorskip("jax")
    h = JAX_PLUGIN.content_hash
    a, a_reexported, b_weights, c_structure = (
        _jax_bytes(0.5),
        _jax_bytes(0.5),
        _jax_bytes(0.9),
        _jax_bytes(0.5, extra_op=True),
    )
    assert a != a_reexported, "jax.export blobs are volatile — the premise of content hashing"
    assert h(a) == h(a_reexported)  # source locations are formatting, not content
    assert h(a) != h(b_weights)
    assert h(a) != h(c_structure)
    validate_plugin(JAX_PLUGIN)


def test_jax_evaluator_matches_jax_exactly() -> None:
    pytest.importorskip("jax")
    from jax import export  # noqa: PLC0415

    payload = _jax_bytes(0.4)
    got = _evaluated(JAX_PLUGIN, payload, [FEATS])
    exported = export.deserialize(bytearray(payload))
    want = np.asarray(exported.call(FEATS.astype("float32").reshape(-1, 1))).reshape(-1)
    assert np.array_equal(got, want.astype("float64"))


# ---------------------------------- Triton (real client objects) ---------------------------------
def test_triton_default_transport_builds_real_client_objects() -> None:
    # no server needed: the DEFAULT transport must construct genuine tritonclient request
    # machinery (the seam a live deployment exercises), and close() must release it.
    pytest.importorskip("tritonclient.http")
    from graphed_preserve.externals import triton_http_transport  # noqa: PLC0415

    client = triton_http_transport({"url": "localhost:8000"})
    try:
        assert hasattr(client, "infer")
    finally:
        client.close()
