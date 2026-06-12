"""The ``xgboost_model`` plugin (M26): XGBoost's open JSON model format; canonical-JSON hash."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import _as_event_array, _canonical_json_hash, _stack_feature_columns


def xgboost_content_hash(payload: bytes) -> str:
    """Canonical JSON of the model document (XGBoost's open format) — dependency-free."""
    return _canonical_json_hash(b"xgboost-json-v1", payload)


def load_xgboost(payload: bytes, params: Mapping[str, Any]) -> Any:
    import xgboost as xgb  # noqa: PLC0415

    booster = xgb.Booster()
    booster.load_model(bytearray(payload))
    return booster


def eval_xgboost(booster: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import xgboost as xgb  # noqa: PLC0415

    return _as_event_array(booster.predict(xgb.DMatrix(_stack_feature_columns(inputs))))


def _xgboost_samples() -> list[bytes]:
    # hash validation needs structurally-plausible distinct JSON documents, not trained models
    return [
        b'{"learner":{"attributes":{},"gradient_booster":{"model":{"trees":[{"split":6.0}]}}},"version":[2,0,0]}',
        b'{"learner":{"attributes":{},"gradient_booster":{"model":{"trees":[{"split":9.5}]}}},"version":[2,0,0]}',
    ]


XGBOOST_PLUGIN = ExternalPlugin(
    kind="xgboost_model",
    content_hash=xgboost_content_hash,
    evaluate=eval_xgboost,
    samples=_xgboost_samples,
    load=load_xgboost,
    framework="xgboost",
)
