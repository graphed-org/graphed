"""graphed-preserve (plan M9): the analysis preservation bundle.

A self-contained, content-addressed export of a graphed analysis that reproduces its histograms
bit-for-bit on a clean machine with no access to the original user code, environment, author, or
input files (inputs are resolved only via the bundle's content-addressed references). Builds on M8's
canonical IR + content-addressed Store; reuses HEP standards (correctionlib / ONNX / UHI) — invents
no formats. Distinct from the M8 Plan: the bundle is the durable *scientific artifact* (A.3.1).
"""

from __future__ import annotations

from .bundle import Bundle, build_bundle, inspect, reproduce
from .errors import PreserveError, UnresolvedPayload
from .externals import eval_correctionlib, eval_onnx, evaluate_external
from .manifest import canonical_bytes, fingerprint

__all__ = [
    "Bundle",
    "PreserveError",
    "UnresolvedPayload",
    "build_bundle",
    "canonical_bytes",
    "eval_correctionlib",
    "eval_onnx",
    "evaluate_external",
    "fingerprint",
    "inspect",
    "reproduce",
]
__version__ = "0.0.1"
