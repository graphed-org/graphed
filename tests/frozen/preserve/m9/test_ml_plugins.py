"""M9 — stress the ExternalPlugin API against real ML-inference frameworks (not just the lambda/toys).

The ``sha256_bytes`` + correctionlib/onnx plugins could be flattering the API. These tests build
plugins around **PyTorch** (TorchScript; hash of weights; single- AND multi-input), **XGBoost**
(booster bytes), and the **NVIDIA Triton** remote-inference pattern (an env-resolved client), to
confirm the ``(content_hash(bytes), evaluate(bytes, params, inputs), samples())`` shape is not
overconfident. Each framework is optional (``pytest.importorskip``); install via ``pip install -e
.[mltest]``. Findings are recorded in ``docs/improvements.rst``.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import (
    Bundle,
    ExternalPlugin,
    build_bundle,
    record_external,
    register_plugin,
    reproduce,
    sha256_bytes,
    validate_plugin,
)

# NOTE: torch + xgboost vendor conflicting OpenMP runtimes; conftest.py sets KMP_DUPLICATE_LIB_OK /
# OMP_NUM_THREADS before any OpenMP library loads so this suite runs both frameworks in one process.

_HIST = {"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0}


# ---- shared record -> build -> reproduce harness ------------------------------------------------
def _hist(value: Any, weight: Any) -> np.ndarray:
    v = np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64")
    return np.histogram(v, bins=_HIST["bins"], range=(_HIST["lo"], _HIST["hi"]), weights=w)[0].round(6)


def _record(plugin: ExternalPlugin, payload: bytes, *, two_inputs: bool = False):  # type: ignore[no-untyped-def]
    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events(n_events=1200, seed=11))
    njet = gak.num(ev.Jet, axis=1)
    feats = [njet, gak.sum(ev.Jet.pt, axis=1)] if two_inputs else [njet]
    weight = record_external(s, plugin, payload, feats)  # the ML model as a preservable External
    return s, ev.MET.pt, weight


def _check_roundtrip(tmp_path, plugin: ExternalPlugin, payload: bytes, *, two_inputs: bool = False) -> Bundle:  # type: ignore[no-untyped-def]
    register_plugin(plugin)  # validates the hash: cross-process determinism + non-vacuity
    s, value, weight = _record(plugin, payload, two_inputs=two_inputs)
    reference = _hist(s.materialize(value), s.materialize(weight))  # build-time, originals present
    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": make_events(n_events=1200, seed=11)},
        payloads={plugin.content_hash(payload): payload},
        histogram=_HIST,
    )
    assert np.array_equal(reproduce(bundle), reference), "from-bundle reproduce != build-time"
    assert bundle.manifest["opaque_nodes"] == [], "a real ML model External must be preserved, not opaque"
    return bundle


# ============================ PyTorch (TorchScript; hash of weights) =============================
def _torch_model_bytes(*, weight: float, bias: float, two_inputs: bool = False) -> bytes:
    import torch  # noqa: PLC0415

    n_in = 2 if two_inputs else 1
    model = torch.nn.Sequential(torch.nn.Linear(n_in, 1), torch.nn.Sigmoid())
    with torch.no_grad():
        model[0].weight.fill_(weight)
        model[0].bias.fill_(bias)
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, n_in))  # self-contained: loadable with no user class
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def _torch_weight_hash(payload: bytes) -> str:
    import torch  # noqa: PLC0415

    model = torch.jit.load(io.BytesIO(payload))
    h = hashlib.sha256(b"torch-weights-v1")
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())  # hash the WEIGHTS, not the zip bytes
    return "sha256:" + h.hexdigest()


def _torch_load(payload: bytes, params: Any) -> Any:
    import torch  # noqa: PLC0415

    model = torch.jit.load(io.BytesIO(payload))  # load the TorchScript module once per worker
    model.eval()
    return model


def _torch_eval(model: Any, params: Any, inputs: list[Any]) -> Any:
    import torch  # noqa: PLC0415

    cols = [np.asarray(ak.to_numpy(ak.Array(i)), dtype="float32") for i in inputs]
    x = torch.from_numpy(np.stack(cols, axis=1))
    with torch.no_grad():
        out = model(x).numpy().reshape(-1)
    return ak.Array(out.astype("float64"))


def _torch_samples() -> list[bytes]:
    return [_torch_model_bytes(weight=0.5, bias=0.0), _torch_model_bytes(weight=0.9, bias=0.1)]


def _torch_samples_2d() -> list[bytes]:
    return [
        _torch_model_bytes(weight=0.3, bias=0.0, two_inputs=True),
        _torch_model_bytes(weight=0.7, bias=0.1, two_inputs=True),
    ]


TORCH_PLUGIN = ExternalPlugin(
    "torch_module", _torch_weight_hash, _torch_eval, _torch_samples, load=_torch_load, framework="torch"
)
TORCH2_PLUGIN = ExternalPlugin(
    "torch_module_2d", _torch_weight_hash, _torch_eval, _torch_samples_2d, load=_torch_load, framework="torch"
)


def test_torch_plugin_hash_is_weights_and_validates() -> None:
    pytest.importorskip("torch")
    validate_plugin(TORCH_PLUGIN)  # deterministic across processes + non-vacuous
    a = _torch_model_bytes(weight=0.5, bias=0.0)
    a_again = _torch_model_bytes(weight=0.5, bias=0.0)
    b = _torch_model_bytes(weight=0.6, bias=0.0)
    # TorchScript zip bytes can vary, but the WEIGHT hash is content-stable and weight-sensitive
    assert _torch_weight_hash(a) == _torch_weight_hash(a_again)
    assert _torch_weight_hash(a) != _torch_weight_hash(b)


def test_torch_model_reproduces_bit_for_bit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    _check_roundtrip(tmp_path, TORCH_PLUGIN, _torch_model_bytes(weight=0.4, bias=-0.2))


def test_torch_multi_input_model_reproduces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # two per-event features -> one external (the API must not assume single-input models)
    pytest.importorskip("torch")
    _check_roundtrip(
        tmp_path, TORCH2_PLUGIN, _torch_model_bytes(weight=0.25, bias=0.0, two_inputs=True), two_inputs=True
    )


# ============================ XGBoost (booster bytes) ============================================
def _xgb_model_bytes(*, seed: int) -> bytes:
    import xgboost as xgb  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 12, size=(300, 1))
    y = (x[:, 0] > 6).astype("float32")
    booster = xgb.train(
        {"max_depth": 2, "objective": "binary:logistic", "seed": 0, "nthread": 1},
        xgb.DMatrix(x, label=y),
        num_boost_round=4,
    )
    return bytes(booster.save_raw("ubj"))  # self-contained model bytes


def _xgb_load(payload: bytes, params: Any) -> Any:
    import xgboost as xgb  # noqa: PLC0415

    booster = xgb.Booster()
    booster.load_model(bytearray(payload))  # load the booster once per worker
    return booster


def _xgb_eval(booster: Any, params: Any, inputs: list[Any]) -> Any:
    import xgboost as xgb  # noqa: PLC0415

    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float32").reshape(-1, 1)
    return ak.Array(booster.predict(xgb.DMatrix(x)).astype("float64"))


def _xgb_samples() -> list[bytes]:
    return [_xgb_model_bytes(seed=1), _xgb_model_bytes(seed=2)]


XGB_PLUGIN = ExternalPlugin(
    "xgboost_model", sha256_bytes, _xgb_eval, _xgb_samples, load=_xgb_load, framework="xgboost"
)


def test_xgboost_plugin_validates_and_reproduces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("xgboost")
    validate_plugin(XGB_PLUGIN)  # the booster bytes ARE the content -> sha256_bytes template suffices
    _check_roundtrip(tmp_path, XGB_PLUGIN, _xgb_model_bytes(seed=3))


# ============================ NVIDIA Triton (remote inference pattern) ===========================
# Triton serves a model REMOTELY: inference is a client.infer() call, and the client/connection is
# part of the *environment*, not the payload. The plugin resolves a client from params (a url) — this
# is the case that probes whether the API is overconfident about externals being pure local bytes.
_TRITON_SERVERS: dict[str, Any] = {}


class _FakeTritonClient:
    """Stands in for a tritonclient connection to a server hosting the served model."""

    def __init__(self, weight: float, bias: float) -> None:
        self.weight, self.bias = weight, bias
        self.closed = False

    def infer(self, model_name: str, x: np.ndarray, real_inputs: Any, real_outputs: Any) -> np.ndarray:
        # real_inputs/real_outputs are genuine tritonclient request objects (built below); a live
        # server would consume them. Offline we compute the served model directly.
        return 1.0 / (1.0 + np.exp(-(self.weight * x[:, 0] + self.bias)))

    def close(self) -> None:
        self.closed = True


def _triton_served_weights(*, weight: float, bias: float) -> bytes:
    return np.array([weight, bias], dtype="float64").tobytes()  # the deployed model's weights


def _triton_connect(payload: bytes, params: Any) -> Any:
    # a live plugin: tritonclient.http.InferenceServerClient(params["url"]); here the env supplies it.
    # Opened ONCE per worker (a connection is exactly the kind of resource `load` exists for).
    return _TRITON_SERVERS[str(params["url"])]


def _triton_disconnect(client: Any) -> None:
    client.close()  # release the connection at the end of the run


def _triton_eval(client: Any, params: Any, inputs: list[Any]) -> Any:
    import tritonclient.http as triton  # noqa: PLC0415

    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float32").reshape(-1, 1)
    # build REAL Triton request objects from our data (validates shapes against the client API)
    inp = triton.InferInput("x", list(x.shape), "FP32")
    inp.set_data_from_numpy(x)
    out = triton.InferRequestedOutput("y")
    # a live run is: client.infer(model, [inp], outputs=[out]); offline the connection computes it.
    result = client.infer(str(params["model"]), x, [inp], [out])
    return ak.Array(np.asarray(result, dtype="float64").reshape(-1))


def _triton_samples() -> list[bytes]:
    return [_triton_served_weights(weight=0.5, bias=0.0), _triton_served_weights(weight=0.8, bias=0.2)]


TRITON_PLUGIN = ExternalPlugin(
    "triton_model",
    sha256_bytes,
    _triton_eval,
    _triton_samples,
    load=_triton_connect,
    close=_triton_disconnect,
    framework="triton",
)


def test_triton_remote_inference_pattern(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("tritonclient.http")
    validate_plugin(TRITON_PLUGIN)  # hash of the served model's weights is deterministic + non-vacuous

    weight, bias, url = 0.45, -0.1, "triton://localhost:8000"
    _TRITON_SERVERS[url] = _FakeTritonClient(weight, bias)  # the connection is part of the environment
    payload = _triton_served_weights(weight=weight, bias=bias)
    register_plugin(TRITON_PLUGIN)

    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events(n_events=1200, seed=11))
    njet = gak.num(ev.Jet, axis=1)
    weight_arr = record_external(s, TRITON_PLUGIN, payload, [njet], params={"url": url, "model": "scorer"})
    value = ev.MET.pt
    reference = _hist(s.materialize(value), s.materialize(weight_arr))

    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=value,
        weight=weight_arr,
        datasets={"events": make_events(n_events=1200, seed=11)},
        payloads={sha256_bytes(payload): payload},
        histogram=_HIST,
    )
    # reproduce succeeds when the Triton server is reachable (here: resolved from the environment).
    assert np.array_equal(reproduce(bundle), reference)
    assert bundle.manifest["opaque_nodes"] == []
    # the served-model identity (weights) IS preserved + content-addressed in the bundle
    (entry,) = bundle.manifest["externals"]
    assert entry["kind"] == "triton_model" and entry["content_hash"] == sha256_bytes(payload)


def test_triton_reproduce_without_the_server_fails_loudly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the honest reproducibility boundary: a remote service must be present in the environment;
    # the bundle preserves the model identity but cannot bottle the server.
    pytest.importorskip("tritonclient.http")
    register_plugin(TRITON_PLUGIN)
    url = "triton://unreachable:8000"
    _TRITON_SERVERS[url] = _FakeTritonClient(0.3, 0.0)
    payload = _triton_served_weights(weight=0.3, bias=0.0)

    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events(n_events=600, seed=5))
    w = record_external(
        s, TRITON_PLUGIN, payload, [gak.num(ev.Jet, axis=1)], params={"url": url, "model": "m"}
    )
    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=ev.MET.pt,
        weight=w,
        datasets={"events": make_events(n_events=600, seed=5)},
        payloads={sha256_bytes(payload): payload},
        histogram=_HIST,
    )
    del _TRITON_SERVERS[url]  # the server goes away (machine B has no access to it)
    with pytest.raises(KeyError):
        reproduce(bundle)
