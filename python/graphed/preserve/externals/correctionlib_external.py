"""The ``correctionlib`` plugin: hash of CONTENTS (canonical JSON), not file bytes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ._base import ExternalPlugin


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
    systematic = str(params.get("systematic", "nominal"))
    x = np.asarray(ak.to_numpy(ak.Array(inputs[0])), dtype="float64")
    return ak.Array(np.asarray(cset[name].evaluate(systematic, x), dtype="float64"))


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
