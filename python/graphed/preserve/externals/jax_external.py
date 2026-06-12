"""The ``jax_export`` plugin (M26): ``jax.export`` artifacts; hash = location-stripped StableHLO + avals."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import _as_event_array, _stack_feature_columns

_MLIR_LOC = re.compile(r"loc\(.*?\)|#loc\d*\s*=?.*")


def jax_content_hash(payload: bytes) -> str:
    """Location-stripped StableHLO + input/output avals — stable across re-exports of one fn."""
    from jax import export  # noqa: PLC0415

    exported = export.deserialize(bytearray(payload))
    h = hashlib.sha256(b"jax-mlir-avals-v1")
    h.update(_MLIR_LOC.sub("", exported.mlir_module()).encode("utf-8"))
    h.update(str(exported.in_avals).encode("utf-8"))
    h.update(str(exported.out_avals).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def load_jax(payload: bytes, params: Mapping[str, Any]) -> Any:
    from jax import export  # noqa: PLC0415

    return export.deserialize(bytearray(payload))


def eval_jax(exported: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    return _as_event_array(exported.call(_stack_feature_columns(inputs)))


def _jax_samples() -> list[bytes]:
    import jax  # noqa: PLC0415
    from jax import export  # noqa: PLC0415

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
)
