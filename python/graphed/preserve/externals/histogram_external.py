"""The ``histogram`` plugin (M25): the fill's canonical spec IS the payload — synthesized
at bundle-build time from the node's own parameters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin


def histogram_content_hash(payload: bytes) -> str:
    """The spec string's SHA-256 — IDENTICAL to the fill node's descriptor hash by construction
    (graphed-histogram derives the node identity from the same canonical spec encoding)."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def eval_histogram(resource: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Reconstruct the fill from the node's own params and run it (M23's evaluator, verbatim)."""
    from graphed_histogram.boost import FillEvaluator  # noqa: PLC0415  (optional integration)

    payload = resource if isinstance(resource, bytes) else bytes(resource)
    evaluator = FillEvaluator(
        spec=payload.decode(),
        n_axes=int(params.get("n_axes", 1)),
        has_weight=bool(params.get("weighted", False)),
        has_sample=bool(params.get("sampled", False)),
    )
    return evaluator(*inputs)


def _histogram_samples() -> list[bytes]:
    # two distinct canonical specs (plain JSON: validating the hash needs no graphed-histogram)
    return [
        b'{"axes":[{"bins":10,"metadata":{},"overflow":true,"start":0.0,"stop":1.0,"type":"Regular","underflow":true}],"storage":"Double","version":1}',
        b'{"axes":[{"bins":20,"metadata":{},"overflow":true,"start":0.0,"stop":2.0,"type":"Regular","underflow":true}],"storage":"Int64","version":1}',
    ]


def _histogram_synthesize(params: Mapping[str, Any]) -> bytes | None:
    spec = params.get("spec")
    return str(spec).encode() if spec is not None else None


HISTOGRAM_PLUGIN = ExternalPlugin(
    kind="histogram",
    content_hash=histogram_content_hash,
    evaluate=eval_histogram,
    samples=_histogram_samples,
    framework="boost_histogram",
    synthesize=_histogram_synthesize,
)
