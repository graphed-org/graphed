"""M28 call templates: a recorded External's ``"$i"`` slots, ``[..]`` groups and constants.

``_ml_matrix`` materializes one entry as the float32 feature matrix an ONNX call takes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np


def _payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return Path(payload).read_bytes()


_TemplateEntry = tuple[str, Any]  # ("slot", i) | ("group", [i, ...]) | ("const", value)


def _parse_template(
    spec: object, n_inputs: int, *, constants: bool, groups: bool
) -> list[_TemplateEntry]:
    if not isinstance(spec, list):
        raise TypeError("a call template's args must be a list (use kwargs= for keywords)")
    entries: list[_TemplateEntry] = []
    for entry in spec:
        if isinstance(entry, str) and entry.startswith("$"):
            i = int(entry[1:])
            if not 0 <= i < n_inputs:
                raise ValueError(f"call template slot {entry} out of range ({n_inputs} inputs)")
            entries.append(("slot", i))
        elif isinstance(entry, list):
            if not groups:
                raise TypeError("this callee takes scalar/array arguments, not stacked groups")
            entries.append(("group", [int(e[1:]) for e in entry]))
        elif constants and isinstance(entry, (str, int, float, bool)):
            entries.append(("const", entry))
        else:
            raise TypeError(
                f"call template entry {entry!r} is not a slot, group, or allowed constant"
            )
    return entries


def _ml_matrix(entry: _TemplateEntry, vals: Sequence[object]) -> object:
    kind, value = entry
    idxs = [value] if kind == "slot" else list(value)
    cols = [np.asarray(ak.to_numpy(ak.Array(vals[i])), dtype="float32") for i in idxs]
    return np.stack(cols, axis=1)
