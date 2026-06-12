"""The ``correctionlib`` plugin: hash of CONTENTS (canonical JSON), not file bytes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
