"""M26 — first-class ML preservation plugins: the registry tier (NO frameworks required).

M9 proved the ``ExternalPlugin`` API shape against ad-hoc in-test plugins; M26 ships the real
thing: registered, exported plugins for **TensorFlow** (``.keras`` archives), **PyTorch**
(TorchScript), **XGBoost** (JSON models), **JAX** (``jax.export`` artifacts), and the **NVIDIA
Triton** remote-inference client. This module pins everything that must hold WITHOUT any ML
framework installed: registration, exports, framework labels, the dependency-free canonical-JSON
hashes (XGBoost, Triton) with their formatting-insensitivity, and full subprocess hash
validation for the two dependency-free plugins.
"""

from __future__ import annotations

import pytest

from graphed_preserve import (
    JAX_PLUGIN,
    PYTORCH_PLUGIN,
    TENSORFLOW_PLUGIN,
    TRITON_PLUGIN,
    XGBOOST_PLUGIN,
    get_plugin,
    validate_plugin,
)

ALL = {
    "tensorflow_model": (TENSORFLOW_PLUGIN, "tensorflow"),
    "pytorch_model": (PYTORCH_PLUGIN, "torch"),
    "xgboost_model": (XGBOOST_PLUGIN, "xgboost"),
    "jax_export": (JAX_PLUGIN, "jax"),
    "triton_model": (TRITON_PLUGIN, "tritonclient"),
}


def test_all_five_plugins_are_registered_at_import() -> None:
    for kind, (plugin, framework) in ALL.items():
        assert get_plugin(kind) is plugin, f"{kind} must be registered by importing graphed_preserve"
        assert plugin.kind == kind
        assert plugin.framework == framework


def test_every_hash_is_namespaced_and_sha256_shaped() -> None:
    # kinds must not collide in hash space even for byte-identical payloads
    blob = b'{"model": "m", "weights": {"w": 0.5, "b": 0.1}}'
    digests = {
        kind: plugin.content_hash(blob)
        for kind, (plugin, _) in ALL.items()
        if kind in ("xgboost_model", "triton_model")
    }
    assert all(d.startswith("sha256:") and len(d) == len("sha256:") + 64 for d in digests.values())
    assert digests["xgboost_model"] != digests["triton_model"]  # domain-separated hashing


@pytest.mark.parametrize("kind", ["xgboost_model", "triton_model"])
def test_canonical_json_hashes_ignore_formatting_but_not_content(kind: str) -> None:
    plugin, _ = ALL[kind]
    a = b'{"model": "scorer", "weights": {"w": 0.5, "b": 0.1}}'
    a_reordered = b'{"weights": {"b": 0.1, "w": 0.5}, "model": "scorer"}'
    a_whitespace = b'{\n  "model":  "scorer",\n  "weights": {"w": 0.5, "b": 0.1}\n}\n'
    b_content = b'{"model": "scorer", "weights": {"w": 0.6, "b": 0.1}}'
    assert plugin.content_hash(a) == plugin.content_hash(a_reordered)  # key order is formatting
    assert plugin.content_hash(a) == plugin.content_hash(a_whitespace)  # whitespace is formatting
    assert plugin.content_hash(a) != plugin.content_hash(b_content)  # values are content


@pytest.mark.parametrize("kind", ["xgboost_model", "triton_model"])
def test_dependency_free_plugins_pass_full_hash_validation(kind: str) -> None:
    # subprocess determinism + non-vacuity, with NO ML framework installed
    validate_plugin(ALL[kind][0])


def test_non_json_payloads_are_rejected_loudly() -> None:
    from graphed_preserve import PreserveError  # noqa: PLC0415

    with pytest.raises(PreserveError, match="JSON"):
        TRITON_PLUGIN.content_hash(b"\x00\x01 not json")
    with pytest.raises(PreserveError, match="JSON"):
        XGBOOST_PLUGIN.content_hash(b"\x89PNG definitely not a model")
