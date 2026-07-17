"""Reference-histogram helpers and content addressing.

Reference outputs are stored as integer bin counts plus a content hash, so a later milestone (or a
re-run on another platform) can assert *bit-for-bit* reproduction. Derived float quantities are
rounded to a fixed precision before cuts/fills (see :func:`stable`) so that binning decisions do
not flip on cross-platform ULP differences — the reference comparison then reduces to exact
integer-count equality.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
from hist import Hist

STABLE_DECIMALS = 6


def stable(values: ak.Array | np.ndarray) -> np.ndarray:
    """Round to a fixed precision for cross-platform-stable cut/bin decisions."""
    return np.round(np.asarray(ak.to_numpy(ak.flatten(values, axis=None))), STABLE_DECIMALS)


def hist1d(values: ak.Array | np.ndarray, *, bins: int, start: float, stop: float, name: str) -> Hist:
    """Fill a 1D integer-count histogram from (already-flat) values."""
    h = Hist.new.Reg(bins, start, stop, name=name).Int64()
    h.fill(stable(values))
    return h


def bin_values(h: Hist) -> list[float]:
    """Histogram bin contents as rounded floats (works for count and weighted histograms)."""
    return [round(float(x), STABLE_DECIMALS) for x in np.asarray(h.values()).ravel()]


def fingerprint(h: Hist) -> str:
    """Stable content hash of a histogram's (rounded) bin contents."""
    payload = json.dumps(bin_values(h), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def reference_record(h: Hist) -> dict[str, object]:
    return {"values": bin_values(h), "fingerprint": fingerprint(h)}


def load_reference(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_reference(path: Path, h: Hist) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference_record(h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
