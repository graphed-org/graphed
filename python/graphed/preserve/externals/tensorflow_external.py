"""The ``tensorflow_model`` plugin (M26): ``.keras`` archives; hash = name-stripped config + weights."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import (
    _as_event_array,
    _stack_feature_columns,
    _strip_config_names,
    ml_matrix,
    parse_call_template,
)


def tensorflow_content_hash(payload: bytes) -> str:
    """Architecture (name-stripped config) + weights — stable across re-saves of one model."""
    model = _keras_load_from_bytes(payload)
    import numpy as np  # noqa: PLC0415

    h = hashlib.sha256(b"keras-config-weights-v1")
    h.update(json.dumps(_strip_config_names(model.get_config()), sort_keys=True, default=str).encode())
    for weights in model.get_weights():
        h.update(np.ascontiguousarray(weights).tobytes())
    return "sha256:" + h.hexdigest()


def _keras_load_from_bytes(payload: bytes) -> Any:
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import keras  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:  # keras loads from a path; fully in-memory after
        path = Path(tmp) / "model.keras"
        path.write_bytes(payload)
        return keras.models.load_model(path, compile=False)


def load_tensorflow(payload: bytes, params: Mapping[str, Any]) -> Any:
    return _keras_load_from_bytes(payload)


def eval_tensorflow(model: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    template = parse_call_template(params, len(inputs))
    if template is None:  # legacy: one stacked feature matrix
        out = model(_stack_feature_columns(inputs))
    else:
        args, kwargs = template
        tensors = [ml_matrix(e, inputs) for e in args]
        kwtensors = {k: ml_matrix(e, inputs) for k, e in kwargs.items()}
        out = model(tensors if len(tensors) > 1 else tensors[0], **kwtensors)
    # convert via the tensor's own .numpy() — np.asarray on an EagerTensor goes through keras's
    # pre-numpy-2 __array__ (no copy= keyword) and warns on every call
    return _as_event_array(out.numpy() if hasattr(out, "numpy") else out)


def _tensorflow_samples() -> list[bytes]:
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import keras  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def _model(weight: float) -> bytes:
        m = keras.Sequential([keras.layers.Input(shape=(1,)), keras.layers.Dense(1, activation="sigmoid")])
        m.layers[0].set_weights([np.array([[weight]], dtype="float32"), np.array([0.0], dtype="float32")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.keras"
            m.save(path)
            return path.read_bytes()

    return [_model(0.5), _model(0.9)]


TENSORFLOW_PLUGIN = ExternalPlugin(
    kind="tensorflow_model",
    content_hash=tensorflow_content_hash,
    evaluate=eval_tensorflow,
    samples=_tensorflow_samples,
    load=load_tensorflow,
    framework="tensorflow",
)
