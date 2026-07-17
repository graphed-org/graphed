"""M9 — reproduce a bundle through a REAL NVIDIA Triton inference server, over gRPC AND HTTP.

This is the fully-realistic counterpart to the in-process Triton pattern in ``test_ml_plugins.py``: a
real ``tritonserver`` container (started by CI) serves the python-backend model in
``tests/samples/triton_models/scorer``; the External plugin opens a real ``tritonclient`` connection
(once per worker, via the ExternalPlugin ``load`` hook) and runs ``client.infer`` for each call. The
served model's identity (its ``model.py`` bytes) is content-addressed in the bundle, and a bundle
reproduced against the live server matches an independent numpy reference **bit-for-bit**.

Gated on ``GRAPHED_TRITON_GRPC`` / ``GRAPHED_TRITON_HTTP`` (set only by the CI ``triton`` job), so
local runs and the normal CI matrix skip it. See ``.github/workflows/ci.yml``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import (
    ExternalPlugin,
    build_bundle,
    record_external,
    register_plugin,
    reproduce,
    sha256_bytes,
)

# the served model's weights — MUST match tests/samples/triton_models/scorer/1/model.py
_W, _B = 0.45, -0.1
_MODEL = "scorer"
_HIST = {"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0}
_SERVERS = {"grpc": os.environ.get("GRAPHED_TRITON_GRPC"), "http": os.environ.get("GRAPHED_TRITON_HTTP")}

# the served model's identity (preserved + content-addressed); inference goes to the live server
_PAYLOAD_PATH = Path(__file__).parents[2] / "samples" / "triton_models" / "scorer" / "1" / "model.py"


def _triton_module(protocol: str) -> Any:
    import importlib  # noqa: PLC0415

    return importlib.import_module(f"tritonclient.{protocol}")


def _triton_connect(payload: bytes, params: Any) -> Any:
    protocol, url = str(params["protocol"]), str(params["url"])
    tc = _triton_module(protocol)
    client = tc.InferenceServerClient(url=url, ssl=False)  # one connection per worker
    for _ in range(60):  # the CI step already waits, but tolerate a slow model load
        try:
            if client.is_server_ready() and client.is_model_ready(str(params["model"])):
                return client
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Triton model {params['model']!r} not ready at {protocol}://{url}")


def _triton_disconnect(client: Any) -> None:
    client.close()


def _triton_infer(client: Any, params: Any, inputs: list[Any]) -> Any:
    tc = _triton_module(str(params["protocol"]))
    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float32").reshape(-1, 1)
    inp = tc.InferInput("x", list(x.shape), "FP32")
    inp.set_data_from_numpy(x)
    out = tc.InferRequestedOutput("y")
    result = client.infer(str(params["model"]), [inp], outputs=[out])
    return ak.Array(result.as_numpy("y").reshape(-1).astype("float64"))


TRITON_SCORER_PLUGIN = ExternalPlugin(
    "triton_scorer",
    content_hash=sha256_bytes,  # the served model's content (its model.py) identifies it in the bundle
    evaluate=_triton_infer,
    samples=lambda: [b"served-model-A", b"served-model-B"],
    load=_triton_connect,  # open the connection once per worker
    close=_triton_disconnect,  # close it at end of run
    framework="triton",
)


def _reference(events: Any) -> np.ndarray:
    """The expected histogram, computed WITHOUT Triton (numpy float32, matching the served model)."""
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    from graphed_awkward import gak  # noqa: PLC0415

    njet = np.asarray(ak.to_numpy(ak.Array(s.materialize(gak.num(ev.Jet, axis=1)))), dtype="float32")
    met = np.asarray(ak.to_numpy(ak.Array(s.materialize(ev.MET.pt))), dtype="float64")
    weight = (1.0 / (1.0 + np.exp(-(_W * njet + _B)))).astype("float32").astype("float64")
    return np.histogram(met, bins=_HIST["bins"], range=(_HIST["lo"], _HIST["hi"]), weights=weight)[0].round(6)


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_real_triton_reproduces_over_transport(protocol: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    url = _SERVERS[protocol]
    if not url:
        pytest.skip(f"no real Triton server (set GRAPHED_TRITON_{protocol.upper()}=host:port)")
    pytest.importorskip(f"tritonclient.{protocol}")
    register_plugin(TRITON_SCORER_PLUGIN)

    payload = _PAYLOAD_PATH.read_bytes()
    events = make_events(n_events=1500, seed=11)
    reference = _reference(events)

    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    njet = gak.num(ev.Jet, axis=1)
    weight = record_external(
        s, TRITON_SCORER_PLUGIN, payload, [njet], params={"protocol": protocol, "url": url, "model": _MODEL}
    )
    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=ev.MET.pt,
        weight=weight,
        datasets={"events": events},
        payloads={sha256_bytes(payload): payload},
        histogram=_HIST,
    )

    # reproduce runs real inference against the live Triton server over this transport
    result = reproduce(bundle)
    assert int(reference.sum()) > 0
    assert np.array_equal(result, reference), f"{protocol}: real-Triton reproduce != numpy reference"
    # the served model is preserved (content-addressed), not opaque
    (entry,) = bundle.manifest["externals"]
    assert entry["kind"] == "triton_scorer" and entry["content_hash"] == sha256_bytes(payload)
