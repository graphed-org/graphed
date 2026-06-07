"""graphed-preserve (plan M9): the analysis preservation bundle.

A self-contained, content-addressed export of a graphed analysis that reproduces its histograms
bit-for-bit on a clean machine with no access to the original user code, environment, author, or
input files (inputs are resolved only via the bundle's content-addressed references). Builds on M8's
canonical IR + content-addressed Store; reuses HEP standards (correctionlib / ONNX / UHI) — invents
no formats. Distinct from the M8 Plan: the bundle is the durable *scientific artifact* (A.3.1).

Externals are an extensible **plugin** system: each plugin gives a ``kind`` a deterministic,
content-based ``content_hash`` (validated for determinism + non-vacuity) and an ``evaluate``. ONNX and
correctionlib ship as plugins and double as templates for users' own Externals.
"""

from __future__ import annotations

from .bundle import Bundle, build_bundle, inspect, reproduce
from .errors import PreserveError, UnresolvedPayload
from .externals import (
    CORRECTIONLIB_PLUGIN,
    ONNX_PLUGIN,
    ExternalPlugin,
    ResourceCache,
    evaluate_external,
    get_plugin,
    record_external,
    register_plugin,
    registered_kinds,
    sha256_bytes,
    validate_plugin,
)
from .manifest import canonical_bytes, fingerprint

__all__ = [
    "CORRECTIONLIB_PLUGIN",
    "ONNX_PLUGIN",
    "Bundle",
    "ExternalPlugin",
    "PreserveError",
    "ResourceCache",
    "UnresolvedPayload",
    "build_bundle",
    "canonical_bytes",
    "evaluate_external",
    "fingerprint",
    "get_plugin",
    "inspect",
    "record_external",
    "register_plugin",
    "registered_kinds",
    "reproduce",
    "sha256_bytes",
    "validate_plugin",
]
__version__ = "0.0.1"
