"""The awkward ``ShuffleBackend`` exchange primitives (plan M39 §3.0, §3.3, §5.1).

``AwkwardBackend`` supplies the vectorized exchange half the generic engine calls:
``partition``/``concat``/``slice_rows``/``estimated_bytes``/``to_wire``/``from_wire`` over awkward
arrays (records with a packed-u64 ``__joinkey__`` column; row payloads may be jagged). Routing is
the pinned §4 rule — ``int.from_bytes(sha256(key.to_bytes(8, "big") [+ salt]).digest()[:8], "big")
% P`` on the join key — process-independent (no Python ``hash()``), so a key lands in the same dest
in every producer process (B2), and the golden vectors pin it against any other pure hash
(ADV-r5.3).
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from typing import Any

import awkward as ak
import numpy as np

#: the pinned §4 route hash (the golden vectors require exactly this).
PINNED_ROUTING_HASH = "sha256"


def route(key: int, parts: int, *, salt: int = 0) -> int:
    """Return the pinned §4/§3.0 route.

    The route is the sha256 of the 8-byte big-endian key (salt=0 == no bytes appended).
    """
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

    Row-conserving; order- and jaggedness-preserving within each dest; deterministic (§4/B2).
    """
    keys = np.asarray(ak.to_numpy(block[key_field])).astype(np.uint64)
    dest = _dests(keys, parts, salt)
    return tuple(block[dest == d] for d in range(parts))


def _unread(layout: Any, length: int) -> Any:
    """``length`` rows of ``layout``'s form with every buffer a placeholder."""
    form, nplike = layout.form, layout.backend.nplike
    placeholders = {
        key: ak.typetracer.PlaceholderArray(nplike, (ak.typetracer.unknown_length,), dtype)
        for key, dtype in form.expected_from_buffers().items()
    }
    return ak.from_buffers(form, length, placeholders, highlevel=False)


def _is_read(layout: Any) -> bool:
    return all(isinstance(b, np.ndarray) for b in ak.to_buffers(layout)[2].values())


def _joined(parts: list[Any]) -> Any:
    """Concatenate layouts of one form row-wise; what a read left unread stays unread.

    Its fields stay too, since a behavior may look for them without reading them.
    """
    if all(_is_read(part) for part in parts):
        return ak.concatenate(parts, axis=0, highlevel=False)
    head, length = parts[0], sum(part.length for part in parts)
    if head.is_record:
        contents = [_joined([p.content(f) for p in parts]) for f in head.fields]
        fields = None if head.is_tuple else head.fields
        return ak.contents.RecordArray(contents, fields, length, parameters=head.parameters)
    if isinstance(head, ak.contents.ListOffsetArray) and all(
        isinstance(p.offsets.data, np.ndarray) for p in parts
    ):
        offsets = [np.asarray(p.offsets.data) for p in parts]
        ends: Any = np.cumsum([o[-1] - o[0] for o in offsets])
        shifts = np.concatenate([[0], ends[:-1]])
        merged = [o[:-1] - o[0] + shift for o, shift in zip(offsets, shifts, strict=True)]
        return ak.contents.ListOffsetArray(
            ak.index.Index64(np.concatenate([*merged, ends[-1:]])),
            _joined([p.content[o[0] : o[-1]] for p, o in zip(parts, offsets, strict=True)]),
            parameters=head.parameters,
        )
    return _unread(head, length)


def concat(blocks: Sequence[ak.Array]) -> ak.Array:
    """Vertically concatenate blocks in order (the deterministic ascending-src_pid merge, §4.0).

    Fields a projected read left as placeholders stay placeholders, of the concatenated length.
    """
    if all(_is_read(block.layout) for block in blocks):
        return ak.concatenate(list(blocks), axis=0)
    head = blocks[0]
    joined = _joined([block.layout for block in blocks])
    return ak.Array(joined, behavior=head.behavior, attrs=head.attrs)


def slice_rows(block: ak.Array, start: int, stop: int) -> ak.Array:
    """Return a half-open row slice at EVENT boundaries.

    Each kept row's list stays intact (§5.1).
    """
    return block[start:stop]


def estimated_bytes(block_or_form: object) -> int:
    """Measured buffer bytes (NOT entry count — jagged bytes are not a function of #rows, §5.1)."""
    return int(getattr(block_or_form, "nbytes", 0))


def to_wire(block: ak.Array) -> bytes:
    """Serialize deterministically via ``ak.to_buffers`` (form + raw buffers).

    The bytes are content-addressed.
    """
    form, length, container = ak.to_buffers(block)
    packed = {k: np.asarray(v) for k, v in container.items()}
    return pickle.dumps((form.to_dict(), int(length), packed), protocol=5)


def from_wire(data: bytes) -> ak.Array:
    """Inverse of :func:`to_wire` — rebuilds values AND jaggedness."""
    form_dict, length, container = pickle.loads(data)
    return ak.from_buffers(ak.forms.from_dict(form_dict), length, container)
