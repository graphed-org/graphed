"""The awkward ``ShuffleBackend`` exchange primitives (plan M39 §3.0, §3.3, §5.1).

``AwkwardBackend`` supplies the vectorized exchange half the generic engine calls:
``partition``/``concat``/``slice_rows``/``estimated_bytes``/``to_wire``/``from_wire`` over awkward
arrays (records with a packed-u64 ``__joinkey__`` column; row payloads may be jagged). Routing is the
pinned §4 rule — ``int.from_bytes(sha256(key.to_bytes(8, "big") [+ salt]).digest()[:8], "big") % P``
on the join key — process-independent (no Python ``hash()``), so a key lands in the same dest in every
producer process (B2), and the golden vectors pin it against any other pure hash (ADV-r5.3).
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence

import awkward as ak
import numpy as np

#: the pinned §4 route hash (the golden vectors require exactly this).
PINNED_ROUTING_HASH = "sha256"


def route(key: int, parts: int, *, salt: int = 0) -> int:
    """The pinned §4/§3.0 route: sha256 of the 8-byte big-endian key (salt=0 == no bytes appended)."""
    key_bytes = int(key).to_bytes(8, "big") + (int(salt).to_bytes(8, "big") if salt else b"")
    return int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "big") % parts


def _dests(keys: np.ndarray, parts: int, salt: int) -> np.ndarray:
    """Per-row destination indices under the pinned route (one sha256 per row)."""
    return np.fromiter(
        (route(int(k), parts, salt=salt) for k in keys), dtype=np.intp, count=len(keys)
    )


def partition(
    block: ak.Array,
    key_field: str,
    parts: int,
    *,
    salt: int = 0,
    boundaries: object = None,
) -> tuple[ak.Array, ...]:
    """Route each row to one of ``parts`` sub-blocks by the pinned hash of ``block[key_field]``.
    Row-conserving; order- and jaggedness-preserving within each dest; deterministic (§4/B2)."""
    keys = np.asarray(ak.to_numpy(block[key_field])).astype(np.uint64)
    dest = _dests(keys, parts, salt)
    return tuple(block[dest == d] for d in range(parts))


def concat(blocks: Sequence[ak.Array]) -> ak.Array:
    """Vertically concatenate blocks in order (the deterministic ascending-src_pid merge, §4.0)."""
    return ak.concatenate(list(blocks), axis=0)


def slice_rows(block: ak.Array, start: int, stop: int) -> ak.Array:
    """A half-open row slice at EVENT boundaries — each kept row's list stays intact (§5.1)."""
    return block[start:stop]


def estimated_bytes(block_or_form: object) -> int:
    """Measured buffer bytes (NOT entry count — jagged bytes are not a function of #rows, §5.1)."""
    return int(getattr(block_or_form, "nbytes", 0))


def to_wire(block: ak.Array) -> bytes:
    """Deterministic serialization via ``ak.to_buffers`` (form + raw buffers) — content-addressed."""
    form, length, container = ak.to_buffers(block)
    packed = {k: np.asarray(v) for k, v in container.items()}
    return pickle.dumps((form.to_dict(), int(length), packed), protocol=5)


def from_wire(data: bytes) -> ak.Array:
    """Inverse of :func:`to_wire` — rebuilds values AND jaggedness."""
    form_dict, length, container = pickle.loads(data)
    return ak.from_buffers(ak.forms.from_dict(form_dict), length, container)
