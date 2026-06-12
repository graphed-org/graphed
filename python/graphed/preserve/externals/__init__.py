"""External payload **plugins** — one module per plugin under this package (see ``_base`` for
the machinery: :class:`ExternalPlugin`, the registry, hash validation, :class:`ResourceCache`).

Importing this package registers every shipped plugin, in a fixed order.
"""

from __future__ import annotations

from ._base import (
    Close,
    ContentHash,
    Evaluate,
    ExternalPlugin,
    Load,
    ResourceCache,
    SynthesizePayload,
    evaluate_external,
    get_plugin,
    record_external,
    register_plugin,
    registered_kinds,
    sha256_bytes,
    validate_plugin,
)
from .correctionlib_external import CORRECTIONLIB_PLUGIN
from .histogram_external import HISTOGRAM_PLUGIN
from .jax_external import JAX_PLUGIN
from .onnx_external import ONNX_PLUGIN
from .pytorch_external import PYTORCH_PLUGIN
from .tensorflow_external import TENSORFLOW_PLUGIN
from .triton_external import TRITON_PLUGIN, triton_http_transport
from .xgboost_external import XGBOOST_PLUGIN

register_plugin(CORRECTIONLIB_PLUGIN, validate=False)
register_plugin(ONNX_PLUGIN, validate=False)
register_plugin(HISTOGRAM_PLUGIN, validate=False)
register_plugin(TENSORFLOW_PLUGIN, validate=False)
register_plugin(PYTORCH_PLUGIN, validate=False)
register_plugin(XGBOOST_PLUGIN, validate=False)
register_plugin(JAX_PLUGIN, validate=False)
register_plugin(TRITON_PLUGIN, validate=False)

__all__ = [
    "CORRECTIONLIB_PLUGIN",
    "HISTOGRAM_PLUGIN",
    "JAX_PLUGIN",
    "ONNX_PLUGIN",
    "PYTORCH_PLUGIN",
    "TENSORFLOW_PLUGIN",
    "TRITON_PLUGIN",
    "XGBOOST_PLUGIN",
    "Close",
    "ContentHash",
    "Evaluate",
    "ExternalPlugin",
    "Load",
    "ResourceCache",
    "SynthesizePayload",
    "evaluate_external",
    "get_plugin",
    "record_external",
    "register_plugin",
    "registered_kinds",
    "sha256_bytes",
    "triton_http_transport",
    "validate_plugin",
]
