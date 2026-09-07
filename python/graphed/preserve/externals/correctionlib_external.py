"""The ``correctionlib`` plugin: hash of CONTENTS (canonical JSON), not file bytes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from ._base import ExternalPlugin
from ._helpers import parse_call_template


def correctionlib_content_hash(payload: bytes) -> str:

    data = json.loads(payload)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(b"correctionlib-contents-v1" + canonical).hexdigest()


def load_correctionlib(payload: bytes, params: Mapping[str, Any]) -> Any:
    """Parse the correction set once (per worker)."""
    import correctionlib  # noqa: PLC0415

    return correctionlib.CorrectionSet.from_string(payload.decode("utf-8"))


def _peel_layout(layout: Any) -> tuple[list[tuple[str, Any]], Any] | None:
    """Split a ListOffset/IndexedOption stack over a 1-D NumpyArray into its layer buffers
    (outer to inner, option layers projected away) and the flat data, or ``None`` if it is
    anything else. Parameters (strings, named records) disqualify: not a numeric column."""
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    layers: list[tuple[str, Any]] = []
    node = layout
    while True:
        if node.parameters:
            return None
        if isinstance(node, ak.contents.NumpyArray):
            data = np.asarray(node.data)
            return (layers, data) if data.ndim == 1 else None
        if isinstance(node, ak.contents.ListOffsetArray):
            offsets = np.asarray(node.offsets.data)
            if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != node.content.length:
                return None  # unreachable head/tail content would be evaluated too
            layers.append(("list", offsets))
            node = node.content
        elif isinstance(node, ak.contents.IndexedOptionArray):
            layers.append(("option", np.asarray(node.index.data)))
            node = node.project()
        else:
            return None


def _same_layers(a: list[tuple[str, Any]], b: list[tuple[str, Any]]) -> bool:
    import numpy as np  # noqa: PLC0415

    return len(a) == len(b) and all(
        ka == kb and np.array_equal(ba, bb) for (ka, ba), (kb, bb) in zip(a, b, strict=True)
    )


def _rebuild(layers: list[tuple[str, Any]], data: Any) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    node: Any = ak.contents.NumpyArray(data)
    for kind, buf in reversed(layers):
        if kind == "list":
            node = ak.contents.ListOffsetArray(ak.index.Index(buf), node)
        else:
            # the projection is ordered by position, so the valid slots renumber 0..n-1
            index = np.full(buf.shape, -1, dtype=np.int64)
            index[buf >= 0] = np.arange(node.length, dtype=np.int64)
            node = ak.contents.IndexedOptionArray(ak.index.Index64(index), node)
    return ak.Array(node)


def _flat_buffer_fast_path(evaluate: Callable[..., Any], call: list[Any]) -> Any | None:
    """Evaluate identically-laid-out array arguments once on their flat buffers.

    correctionlib's awkward wrapper broadcasts and ``ak.transform``s every call; that dominates a
    jagged option-typed evaluation and is redundant when every array argument already shares one
    layout. The guard is structural equality of every offsets/index buffer, so the values, the None
    mask and the type of the rebuilt result are the ones the wrapper would have produced. ``None``
    means "not that shape" — the caller falls back to the dispatched call."""
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    layers: list[tuple[str, Any]] | None = None
    flat: list[Any] = []
    n: int | None = None
    for value in call:
        if isinstance(value, ak.Array):
            peeled = _peel_layout(value.layout)
            if peeled is None or (layers is not None and not _same_layers(layers, peeled[0])):
                return None
            layers, data = peeled
            n = data.shape[0]
            flat.append(data)
        elif isinstance(value, (str, int, float, np.generic)):
            flat.append(value)
        else:
            return None
    if layers is None:
        return None
    out = np.asarray(evaluate(*flat), dtype="float64")
    if out.ndim != 1 or out.shape[0] != n:
        return None
    return _rebuild(layers, out)


def eval_correctionlib(cset: Any, params: Mapping[str, Any], inputs: list[Any]) -> Any:
    import awkward as ak  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    name = str(params.get("name", ""))
    template = parse_call_template(
        params, len(inputs), allow_constants=True, allow_groups=False, allow_kwargs=False
    )
    if template is None:  # the legacy (systematic, inputs[0]) shape, unchanged
        systematic = str(params.get("systematic", "nominal"))
        x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float64")
        return ak.Array(np.asarray(cset[name].evaluate(systematic, x), dtype="float64"))
    args, _ = template
    # correctionlib accepts numpy AND awkward natively (jagged included) — pass inputs through
    call = [inputs[v] if kind == "slot" else v for kind, v in args]
    fast = _flat_buffer_fast_path(cset[name].evaluate, call)
    if fast is not None:
        return fast
    out = cset[name].evaluate(*call)
    return out if isinstance(out, ak.Array) else ak.Array(np.asarray(out))


def _correctionlib_samples() -> list[bytes]:
    def _cset(sf: float) -> bytes:

        doc = {
            "schema_version": 2,
            "corrections": [
                {
                    "name": "sf",
                    "version": 1,
                    "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                    "output": {"name": "sf", "type": "real"},
                    "data": {
                        "nodetype": "category",
                        "input": "systematic",
                        "content": [{"key": "nominal", "value": sf}],
                    },
                }
            ],
        }
        return json.dumps(doc).encode("utf-8")

    return [_cset(1.0), _cset(1.5)]


CORRECTIONLIB_PLUGIN = ExternalPlugin(
    kind="correctionlib",
    content_hash=correctionlib_content_hash,
    evaluate=eval_correctionlib,
    samples=_correctionlib_samples,
    load=load_correctionlib,  # parse the correction set once per worker
    framework="correctionlib",
)
