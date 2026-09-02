"""The ``histogram`` plugin (M25): the fill's canonical spec IS the payload — synthesized
at bundle-build time from the node's own parameters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin


def histogram_content_hash(payload: bytes) -> str:
    """The payload string's SHA-256 — IDENTICAL to the fill node's descriptor hash by construction
    (graphed-histogram derives the node identity from the same canonical encoding; see
    :func:`_histogram_synthesize` for the discriminated form H1 folds in)."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def eval_histogram(resource: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    """Reconstruct the fill from the node's OWN params and run it (M23's evaluator, verbatim).

    The payload is only a content-address WITNESS (see :func:`_histogram_synthesize`): the spec and
    the evaluator's shape both ride in ``params``, so replay never decodes the (now discriminated)
    payload bytes. Reconstructing the real :class:`FillEvaluator` — with its ``n_weights`` and
    §6.2 ``variation`` — lets it fold multiple weights and run the axis-mode loop itself."""
    del resource  # the payload is a hash witness; the fill is rebuilt from params
    from graphed_histogram.boost import FillEvaluator  # noqa: PLC0415  (optional integration)

    variation = params.get("variation")
    evaluator = FillEvaluator(
        spec=str(params["spec"]),
        n_axes=int(params.get("n_axes", 1)),
        has_weight=bool(params.get("weighted", False)),
        has_sample=bool(params.get("sampled", False)),
        n_weights=int(params.get("n_weights", 1)),
        variation=tuple(json.loads(variation)) if variation is not None else None,
    )
    return evaluator(*inputs)


def _histogram_samples() -> list[bytes]:
    # two distinct canonical specs (plain JSON: validating the hash needs no graphed-histogram)
    return [
        b'{"axes":[{"bins":10,"metadata":{},"overflow":true,"start":0.0,"stop":1.0,"type":"Regular","underflow":true}],"storage":"Double","version":1}',
        b'{"axes":[{"bins":20,"metadata":{},"overflow":true,"start":0.0,"stop":2.0,"type":"Regular","underflow":true}],"storage":"Int64","version":1}',
    ]


def _canonical_payload(params: Mapping[str, Any]) -> str:
    """graphed-histogram ``boost._fill_chash``'s canonical string, rebuilt from the node's params:
    the bare spec for the canonical single-weight sibling fill, else the spec with a
    ``\\x00``-joined discriminator suffix (``unweighted`` / ``n_weights=N`` / ``variation=<json>``)
    folded in — the SAME set and order ``_fill_chash`` uses, so re-hashing it reproduces the
    recorded node id. ``params["variation"]`` is already ``json.dumps(list(node_labels))``."""
    spec = str(params["spec"])
    disc: list[str] = []
    if not bool(params.get("weighted", False)):
        disc.append("unweighted")
    n_weights = int(params.get("n_weights", 1))
    if n_weights != 1:
        disc.append(f"n_weights={n_weights}")
    variation = params.get("variation")
    if variation is not None:
        disc.append("variation=" + str(variation))
    return spec if not disc else spec + "\x00" + "\x00".join(disc)


def _histogram_synthesize(params: Mapping[str, Any], recorded_hash: str) -> bytes | None:
    """Synthesize the fill's payload bytes from its params (M25). The payload is a content-address
    WITNESS only — replay rebuilds the fill from params (:func:`eval_histogram`) — so it just has to
    hash to the recorded node id. TWO record paths mint that id from identical params: H1's
    ``Histogram.fill`` (the discriminated ``_fill_chash``) and the legacy ``record_external`` over
    the bare spec bytes. Only the recorded hash tells them apart — emit whichever form matches it."""
    if params.get("spec") is None:
        return None
    for candidate in (_canonical_payload(params), str(params["spec"])):
        if histogram_content_hash(candidate.encode()) == recorded_hash:
            return candidate.encode()
    # neither form matches: a genuinely mismatched/poisoned id — return the canonical form so
    # build_bundle's integrity check surfaces it (never silently store under a wrong id)
    return _canonical_payload(params).encode()


HISTOGRAM_PLUGIN = ExternalPlugin(
    kind="histogram",
    content_hash=histogram_content_hash,
    evaluate=eval_histogram,
    samples=_histogram_samples,
    framework="boost_histogram",
    synthesize=_histogram_synthesize,
)
