"""Shared helpers for the ML-framework plugins (M26)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..errors import PreserveError


def _canonical_json_hash(domain: bytes, payload: bytes) -> str:
    """sha256 over domain-separated, canonicalized JSON (key order + whitespace = formatting)."""
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise PreserveError(f"payload is not valid JSON ({err})") from err
    canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(domain + canon.encode("utf-8")).hexdigest()


def _stack_feature_columns(inputs: list[Any]) -> Any:
    """Per-event feature arrays -> one float32 (n_events, n_features) matrix (the M9 convention)."""
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cols = [np.asarray(ak.to_numpy(ak.Array(i)), dtype="float32") for i in inputs]
    return np.stack(cols, axis=1)


def _as_event_array(out: Any) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    return ak.Array(np.asarray(out, dtype="float64").reshape(-1))


def _strip_config_names(obj: Any) -> Any:
    """Drop auto-generated layer ``name`` entries from a keras config: incidental, not content."""
    if isinstance(obj, dict):
        return {k: _strip_config_names(v) for k, v in obj.items() if k != "name"}
    if isinstance(obj, list):
        return [_strip_config_names(v) for v in obj]
    return obj
