"""The ``jax_export`` plugin (M26): ``jax.export`` artifacts.

The hash is location-stripped StableHLO + avals.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import (
    RowShape,
    _as_event_array,
    _stack_feature_columns,
    memoized_model_hash,
    ml_matrix,
    parse_call_template,
    row_shape,
)

_MLIR_LOC = re.compile(r"loc\(.*?\)|#loc\d*\s*=?.*")


def jax_content_hash(payload: bytes) -> str:
    """Return the content hash of a ``jax.export`` payload."""
    return memoized_model_hash("jax", payload, _jax_identity)[0]


def jax_output_shape(payload: bytes) -> RowShape:
    """Return the row shape of the export's first output, as its avals state it."""
    return memoized_model_hash("jax", payload, _jax_identity)[1]


def _jax_identity(payload: bytes) -> tuple[str, RowShape]:
    """Location-stripped StableHLO + input/output avals — stable across re-exports of one fn."""
    from jax import export

    exported = export.deserialize(bytearray(payload))
    h = hashlib.sha256(b"jax-mlir-avals-v1")
    h.update(_MLIR_LOC.sub("", exported.mlir_module()).encode("utf-8"))
    h.update(str(exported.in_avals).encode("utf-8"))
    h.update(str(exported.out_avals).encode("utf-8"))
    shape = row_shape(exported.out_avals[0].shape) if exported.out_avals else None
    return "sha256:" + h.hexdigest(), shape


def load_jax(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Deserialize a ``jax.export`` payload."""
    from jax import export

    return export.deserialize(bytearray(payload))


def eval_jax(exported: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Call the exported function on ``inputs``, one value per event."""
    import numpy as np

    template = parse_call_template(params, len(inputs))
    if template is None:  # legacy: one stacked feature matrix
        arrays, kwarrays = [_stack_feature_columns(inputs)], {}
    else:
        args, kwargs = template
        arrays = [ml_matrix(e, inputs) for e in args]
        kwarrays = {k: ml_matrix(e, inputs) for k, e in kwargs.items()}
    if not any(len(a) for a in (*arrays, *kwarrays.values())):
        tail = row_shape(exported.out_avals[0].shape)
        # the export states the width; a symbolic batch dimension refuses zero rows
        if tail is not None:
            return _as_event_array(np.zeros((0, *tail)))
    return _as_event_array(exported.call(*arrays, **kwarrays))


def _jax_samples() -> list[bytes]:
    import jax
    from jax import export

    def _artifact(weight: float) -> bytes:
        def f(x: Any) -> Any:
            return jax.nn.sigmoid(weight * x[:, 0] + 0.1)

        spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 1"), jax.numpy.float32)
        return bytes(export.export(jax.jit(f))(spec).serialize())

    return [_artifact(0.5), _artifact(0.9)]


JAX_PLUGIN = ExternalPlugin(
    kind="jax_export",
    content_hash=jax_content_hash,
    evaluate=eval_jax,
    samples=_jax_samples,
    load=load_jax,
    framework="jax",
    output_dtype="float64",
    output_shape=jax_output_shape,
)
