"""The ``xgboost_model`` plugin (M26): XGBoost's open JSON model format; canonical-JSON hash."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import PreserveError
from ._base import ExternalPlugin
from ._helpers import (
    RowShape,
    _as_event_array,
    _stack_feature_columns,
    canonical_json_identity,
    json_bytes,
    ml_matrix,
    parse_call_template,
)


def xgboost_content_hash(payload: bytes) -> str:
    """Canonical JSON of the model document (XGBoost's open format) — dependency-free."""
    return canonical_json_identity(b"xgboost-json-v1", payload, _row_shape)[0]


def xgboost_output_shape(payload: bytes) -> RowShape:
    """Return the row shape ``predict`` gives, as the model's objective and params state it."""
    return canonical_json_identity(b"xgboost-json-v1", payload, _row_shape)[1]


def _row_shape(doc: Any) -> RowShape:
    try:
        learner = doc["learner"]
        objective, model = learner["objective"]["name"], learner["learner_model_param"]
        if objective == "multi:softprob":
            return (int(model["num_class"]),)
        # before 1.6 a model had no num_target, and xgboost loads it as 1
        return (1,) if objective == "multi:softmax" else (int(model.get("num_target", 1)),)
    except (KeyError, TypeError, ValueError):
        return None


def load_xgboost(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Load an XGBoost booster from its JSON payload."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(json_bytes(payload)))
    return booster


def eval_xgboost(booster: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Predict with the booster over ``inputs``, one value per event."""
    template = parse_call_template(params, len(inputs), allow_kwargs=False)
    if template is not None:
        args, _ = template
        if len(args) != 1 or args[0][0] not in ("group", "slot"):
            raise PreserveError(
                "xgboost models take a single tabular matrix: the call template must be "
                'exactly one positional group (e.g. [["$0", "$1"]])'
            )
        matrix = ml_matrix(args[0], inputs)
    else:
        matrix = _stack_feature_columns(inputs)
    import numpy as np
    import xgboost as xgb

    # xgboost reads absent trailing columns as missing values instead of refusing them
    expected = booster.num_features()
    if matrix.shape[1] != expected:
        raise PreserveError(
            f"the xgboost model takes {expected} features; the call gives {matrix.shape[1]}"
        )
    names = booster.feature_names  # its own, so a model trained on named columns runs as shipped
    if not len(matrix):  # an empty DMatrix predicts shape (0, 0), losing the output width
        one = booster.predict(xgb.DMatrix(np.zeros((1, expected)), feature_names=names))
        return _as_event_array(one[:0])
    return _as_event_array(booster.predict(xgb.DMatrix(matrix, feature_names=names)))


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
    output_dtype="float64",
    output_shape=xgboost_output_shape,
)
