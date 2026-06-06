"""M9 — the External plugin system: validated deterministic hashes + a user-defined template.

Plan M9 (this refactor): Externals are any user-provided function with a deterministic content hash.
The hash is checked for determinism (across processes) and non-vacuity; ONNX hashes its weights and
correctionlib its contents (not the raw file bytes). Users follow the same shape for their own kinds.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_corpus import make_events

from graphed_preserve import (
    CORRECTIONLIB_PLUGIN,
    ONNX_PLUGIN,
    Bundle,
    ExternalPlugin,
    PreserveError,
    build_bundle,
    inspect,
    record_external,
    register_plugin,
    reproduce,
    validate_plugin,
)


# ---- the built-in plugins pass validation -------------------------------------------------------
def test_builtin_plugins_have_valid_hashes() -> None:
    validate_plugin(CORRECTIONLIB_PLUGIN)  # deterministic across processes + non-vacuous
    validate_plugin(ONNX_PLUGIN)


def test_correctionlib_hashes_contents_not_formatting() -> None:
    a = json.dumps({"schema_version": 2, "x": 1}, indent=4).encode()
    b = json.dumps({"x": 1, "schema_version": 2}, separators=(",", ":")).encode()  # same content, diff bytes
    assert a != b
    assert CORRECTIONLIB_PLUGIN.content_hash(a) == CORRECTIONLIB_PLUGIN.content_hash(b)
    c = json.dumps({"schema_version": 2, "x": 2}).encode()  # different content
    assert CORRECTIONLIB_PLUGIN.content_hash(a) != CORRECTIONLIB_PLUGIN.content_hash(c)


def test_onnx_hashes_weights() -> None:
    import agc  # noqa: PLC0415

    m1 = agc.onnx_model(weight=0.1)
    m1_again = agc.onnx_model(weight=0.1)
    m2 = agc.onnx_model(weight=0.2)
    assert ONNX_PLUGIN.content_hash(m1) == ONNX_PLUGIN.content_hash(m1_again)  # weights identical
    assert ONNX_PLUGIN.content_hash(m1) != ONNX_PLUGIN.content_hash(m2)  # weights differ


# ---- validation rejects bad hashes --------------------------------------------------------------
def _constant_hash(payload: bytes) -> str:
    return "sha256:constant"  # vacuous: ignores the payload


def _salted_hash(payload: bytes) -> str:
    return f"sha256:{hash(payload) & 0xFFFFFFFF:x}"  # builtin hash() is salted per process


def _two_samples() -> list[bytes]:
    return [b"alpha", b"beta"]


def test_vacuous_hash_is_rejected() -> None:
    plugin = ExternalPlugin(
        kind="vac", content_hash=_constant_hash, evaluate=lambda *_: None, samples=_two_samples
    )
    with pytest.raises(PreserveError, match="vacuous"):
        register_plugin(plugin)


def test_nondeterministic_hash_is_rejected() -> None:
    plugin = ExternalPlugin(
        kind="salt", content_hash=_salted_hash, evaluate=lambda *_: None, samples=_two_samples
    )
    with pytest.raises(PreserveError, match="deterministic across processes"):
        register_plugin(plugin)


def test_time_based_hash_is_rejected() -> None:
    plugin = ExternalPlugin(
        kind="clock",
        content_hash=lambda payload: f"sha256:{int(time.time() * 1e6)}",
        evaluate=lambda *_: None,
        samples=_two_samples,
    )
    with pytest.raises(PreserveError):
        register_plugin(plugin)


# ---- a user-defined plugin, end to end (the template users follow) ------------------------------
def _linear_hash(payload: bytes) -> str:
    cfg = json.loads(payload)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(b"linear-sf-v1" + canonical).hexdigest()


def _linear_eval(payload: bytes, params: Any, inputs: list[Any]) -> Any:
    cfg = json.loads(payload)
    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float64")
    return ak.Array(cfg["slope"] * x + cfg["intercept"])


def _linear_samples() -> list[bytes]:
    return [
        json.dumps({"slope": 1.0, "intercept": 0.0}).encode(),
        json.dumps({"slope": 2.0, "intercept": 1.0}).encode(),
    ]


LINEAR_PLUGIN = ExternalPlugin(
    kind="linear_sf",
    content_hash=_linear_hash,
    evaluate=_linear_eval,
    samples=_linear_samples,
    framework="demo",
)


def _record_user_analysis(payload: bytes):  # type: ignore[no-untyped-def]
    from graphed_awkward import gak  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events(n_events=1500, seed=7))
    njet = gak.num(ev.Jet, axis=1)
    weight = record_external(s, LINEAR_PLUGIN, payload, [njet])  # user-defined External
    return s, ev.MET.pt, weight


def test_user_plugin_registers_and_reproduces_bit_for_bit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    register_plugin(LINEAR_PLUGIN)  # validated: deterministic across processes + non-vacuous
    payload = json.dumps({"slope": 1.5, "intercept": 0.25}).encode()
    s, value, weight = _record_user_analysis(payload)
    reference = np.histogram(
        np.asarray(ak.to_numpy(ak.Array(s.materialize(value))), dtype="float64"),
        bins=20,
        range=(0.0, 200.0),
        weights=np.asarray(ak.to_numpy(ak.Array(s.materialize(weight))), dtype="float64"),
    )[0].round(6)

    bundle = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": make_events(n_events=1500, seed=7)},
        payloads={LINEAR_PLUGIN.content_hash(payload): payload},
        histogram={"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0},
    )
    assert np.array_equal(reproduce(bundle), reference)  # user External reproduces from the bundle
    # the user External is preserved (durable), not flagged opaque, and shown by inspect
    assert bundle.manifest["opaque_nodes"] == []
    assert "linear_sf" in inspect(bundle)


def test_user_plugin_fingerprint_tracks_its_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    register_plugin(LINEAR_PLUGIN)

    def _fp(slope: float) -> str:
        payload = json.dumps({"slope": slope, "intercept": 0.0}).encode()
        s, value, weight = _record_user_analysis(payload)
        b = build_bundle(
            tmp_path / f"b{slope}",
            session=s,
            value=value,
            weight=weight,
            datasets={"events": make_events(n_events=1500, seed=7)},
            payloads={LINEAR_PLUGIN.content_hash(payload): payload},
            histogram={"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0},
        )
        return b.fingerprint()

    assert _fp(1.0) == _fp(1.0)  # deterministic
    assert _fp(1.0) != _fp(2.0)  # changing the user payload changes the bundle fingerprint


def test_reopened_user_bundle_reproduces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    register_plugin(LINEAR_PLUGIN)
    payload = json.dumps({"slope": 0.7, "intercept": 0.1}).encode()
    s, value, weight = _record_user_analysis(payload)
    built = build_bundle(
        tmp_path / "bundle",
        session=s,
        value=value,
        weight=weight,
        datasets={"events": make_events(n_events=1500, seed=7)},
        payloads={LINEAR_PLUGIN.content_hash(payload): payload},
        histogram={"name": "met", "bins": 20, "lo": 0.0, "hi": 200.0},
    )
    reopened = Bundle.open(built.root)  # a fresh handle (as machine B would open it)
    assert np.array_equal(reproduce(reopened), reproduce(built))
