"""M26 — end-to-end: ML externals recorded into an analysis, preserved, reproduced bit-for-bit.

Two tiers again. The **Triton** round trip runs with NO frameworks at all (the injectable
``transport`` ref resolves the suite's fake — the exact mechanism a live deployment uses with
``tritonclient``), so the full record→build→reproduce path for a *remote* model is exercised on
every CI cell. It also pins the honest reproducibility boundary: the bundle preserves the served
model's identity, but a vanished server fails LOUDLY at reproduce time. The **XGBoost** round
trip (lightest real framework) pins the same path for a *local-bytes* model.
"""

from __future__ import annotations

import os
from typing import Any

import awkward as ak
import fake_triton
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import (
    TRITON_PLUGIN,
    XGBOOST_PLUGIN,
    build_bundle,
    record_external,
    reproduce,
)

HIST = {"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0}


def _hist(value: Any, weight: Any) -> np.ndarray:
    v = np.asarray(ak.to_numpy(ak.Array(value)), dtype="float64")
    w = np.asarray(ak.to_numpy(ak.Array(weight)), dtype="float64")
    return np.histogram(v, bins=HIST["bins"], range=(HIST["lo"], HIST["hi"]), weights=w)[0].round(
        6
    )  # the triple-path _histogram convention


def _record(plugin: Any, payload: bytes, params: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events(n_events=1200, seed=11))
    njet = gak.num(ev.Jet, axis=1)
    weight = record_external(s, plugin, payload, [njet], params=params or {})
    return s, ev.MET.pt, weight


def _roundtrip(tmp_path, plugin: Any, payload: bytes, params: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    s, value, weight = _record(plugin, payload, params)
    reference = _hist(s.materialize(value), s.materialize(weight))
    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": make_events(n_events=1200, seed=11)},
        payloads={plugin.content_hash(payload): payload},
        histogram=HIST,
    )
    assert np.array_equal(np.asarray(reproduce(bundle), dtype="float64"), reference)
    assert bundle.manifest["opaque_nodes"] == []  # a first-class ML model is never opaque
    (entry,) = bundle.manifest["externals"]
    assert entry["kind"] == plugin.kind
    assert entry["content_hash"] == plugin.content_hash(payload)
    return bundle


TRITON_DESCRIPTOR = b'{"model": "scorer", "version": "1", "weights": {"w": 0.45, "b": -0.1}, "io": {"input": "x", "output": "y"}}'
TRITON_PARAMS = {
    "url": "triton://demo:8000",
    "model": "scorer",
    "transport": "fake_triton:transport",  # the injectable seam; a live run omits this (tritonclient default)
    "input_name": "x",
    "output_name": "y",
}


def test_triton_external_preserves_and_reproduces_without_any_framework(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = fake_triton.serve(TRITON_PARAMS["url"], TRITON_DESCRIPTOR)
    _roundtrip(tmp_path, TRITON_PLUGIN, TRITON_DESCRIPTOR, TRITON_PARAMS)
    assert client.infer_calls > 0  # the inference genuinely went through the transport
    assert client.closed  # ... and the connection was released at end of run


def test_triton_reproduce_with_the_server_gone_fails_loudly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    url = "triton://gone:8000"
    params = dict(TRITON_PARAMS, url=url)
    fake_triton.serve(url, TRITON_DESCRIPTOR)
    bundle = _roundtrip(tmp_path, TRITON_PLUGIN, TRITON_DESCRIPTOR, params)
    del fake_triton.SERVERS[url]  # machine B cannot reach the server
    with pytest.raises(Exception, match=r"gone:8000|KeyError|triton") as exc:
        reproduce(bundle)
    assert "triton://gone:8000" in str(exc.value) or isinstance(exc.value, KeyError)


def test_triton_unimportable_transport_fails_loudly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from graphed_preserve import PreserveError  # noqa: PLC0415

    params = dict(TRITON_PARAMS, transport="no_such_module:transport")
    with pytest.raises((PreserveError, ImportError, ModuleNotFoundError)):
        TRITON_PLUGIN.load(TRITON_DESCRIPTOR, params)


def test_xgboost_external_preserves_and_reproduces_bit_for_bit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 12, size=(300, 1))
    y = (x[:, 0] > 6).astype("float32")
    booster = xgb.train(
        {"max_depth": 2, "objective": "binary:logistic", "seed": 0, "nthread": 1},
        xgb.DMatrix(x, label=y),
        num_boost_round=4,
    )
    _roundtrip(tmp_path, XGBOOST_PLUGIN, bytes(booster.save_raw("json")))


# ---------------------------------- Triton, LIVE (CI only) ---------------------------------------
LIVE_TRITON_URL = os.environ.get("TRITON_SERVER_URL", "")
LIVE_DESCRIPTOR = b'{"model": "scorer", "version": "1", "weights": {"w": 0.45, "b": -0.1}, "io": {"input": "x", "output": "y"}}'


@pytest.mark.skipif(
    not LIVE_TRITON_URL,
    reason="needs a live Triton server (set TRITON_SERVER_URL; the CI test-triton-live job runs this)",
)
def test_triton_live_server_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The DEFAULT tritonclient transport against a real running Triton server (no fakes, no
    transport override): record -> build -> reproduce with every inference over the wire, and
    the result independently checked against the served model's declared weights."""
    pytest.importorskip("tritonclient.http")
    from graphed_preserve import TRITON_PLUGIN as plugin  # noqa: PLC0415

    params = {"url": LIVE_TRITON_URL, "model": "scorer", "input_name": "x", "output_name": "y"}

    # independent check first: one direct evaluation against sigmoid(w*x + b) from the
    # descriptor (the served identity) — FP32 over the wire, so allclose not array_equal
    feats = np.array([0.0, 1.0, 2.5, -1.0, 4.0], dtype="float64")
    resource = plugin.load(LIVE_DESCRIPTOR, params)
    try:
        got = np.asarray(ak.to_numpy(ak.Array(plugin.evaluate(resource, params, [feats]))))
    finally:
        plugin.close(resource)
    want = 1.0 / (1.0 + np.exp(-(0.45 * feats - 0.1)))
    assert np.allclose(got, want, atol=1e-6)

    # the full preservation round trip THROUGH THE SERVER, bit-for-bit build vs reproduce
    _roundtrip(tmp_path, plugin, LIVE_DESCRIPTOR, params)
