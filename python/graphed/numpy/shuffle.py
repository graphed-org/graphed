"""The numpy ``ShuffleBackend`` exchange primitives (plan M39 §3.0, §3.3, B-r5.3).

A SECOND exchange backend ships in M39 (not M40) so the generic engine's backend-agnosticism is
witnessed by EXECUTION, not merely the §A.4 import lint. numpy's rectilinear primitives are trivial
— ``partition``/``concat``/``slice_rows``/``estimated_bytes``/``to_wire``/``from_wire`` over
structured (record) arrays — but they obey the SAME pinned §4 routing rule the awkward backend does,
proven by the shared golden vectors (a numpy impl using any other hash fails them identically).

The route is ``int.from_bytes(sha256(key.to_bytes(8, "big") [+ salt]).digest()[:8], "big") % P`` on
the packed-u64 ``__joinkey__`` column — process-independent (no Python ``hash()``, which is
PYTHONHASHSEED-salted) so the same key lands in the same dest in every producer process (B2).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import numpy as np

#: the pinned §4 route hash (the golden vectors require exactly this; measured in exec-local's
#: benchmark against a non-crypto alternative).
PINNED_ROUTING_HASH = "sha256"


def route(key: int, parts: int, *, salt: int = 0) -> int:
    """The pinned §4/§3.0 route: sha256 of the 8-byte big-endian key (salt=0 == no bytes appended)."""
    key_bytes = int(key).to_bytes(8, "big") + (int(salt).to_bytes(8, "big") if salt else b"")
    return int.from_bytes(hashlib.sha256(key_bytes).digest()[:8], "big") % parts


def _dests(keys: np.ndarray, parts: int, salt: int) -> np.ndarray:
    """Per-row destination indices under the pinned route (one sha256 per row)."""
    return np.fromiter((route(int(k), parts, salt=salt) for k in keys), dtype=np.intp, count=len(keys))


def partition(
    block: np.ndarray,
    key_field: str,
    parts: int,
    *,
    salt: int = 0,
    boundaries: object = None,
) -> tuple[np.ndarray, ...]:
    """Route each record to one of ``parts`` sub-blocks by the pinned hash of ``block[key_field]``.
    Row-conserving and order-preserving within each dest; deterministic (§4/B2)."""
    dest = _dests(np.asarray(block[key_field]).astype(np.uint64), parts, salt)
    return tuple(block[dest == d] for d in range(parts))


def concat(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Vertically concatenate record blocks in order (the ascending-src_pid merge). The engine only
    ever merges a dest's >=1 contributing blocks, so an empty list is a caller error (np raises)."""
    return np.concatenate([np.asarray(b) for b in blocks])


def slice_rows(block: np.ndarray, start: int, stop: int) -> np.ndarray:
    """A contiguous half-open record slice."""
    return np.asarray(block)[start:stop]


def estimated_bytes(block_or_form: object) -> int:
    """Measured payload bytes: the record array's ``nbytes`` (scales with itemsize)."""
    return int(np.asarray(block_or_form).nbytes)


def to_wire(block: np.ndarray) -> bytes:
    """Deterministic ``.npy`` serialization (header + raw bytes) — content-addressing hashes these."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(block), allow_pickle=False)
    return buf.getvalue()


def from_wire(data: bytes) -> np.ndarray:
    """Inverse of :func:`to_wire`."""
    return np.load(io.BytesIO(data), allow_pickle=False)
