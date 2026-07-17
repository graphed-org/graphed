"""M39 — the numpy ``ShuffleBackend`` exchange primitives (plan §3.0, §3.3, B-r5.3).

A SECOND exchange backend ships in M39 (not M40) so the generic engine's backend-agnosticism is
witnessed by execution, not just the §A.4 import lint. numpy's rectilinear primitives are trivial —
``partition``/``concat``/``slice_rows``/``estimated_bytes``/``to_wire``/``from_wire`` over structured
(record) arrays — but they must obey the SAME pinned §4 routing rule, proven by the golden vectors
(a numpy impl using any other hash fails them exactly as the awkward one would).
"""

from __future__ import annotations

import numpy as np
from golden_route import GOLDEN

from graphed.numpy import NumpyBackend

_DTYPE = np.dtype([("__joinkey__", np.uint64), ("v", np.int64)])


def _single_key_block(key: int, rows: int = 3) -> np.ndarray:
    block = np.zeros(rows, dtype=_DTYPE)
    block["__joinkey__"] = np.uint64(key)
    block["v"] = np.arange(rows, dtype=np.int64)
    return block


def _block(keys: list[int]) -> np.ndarray:
    block = np.zeros(len(keys), dtype=_DTYPE)
    block["__joinkey__"] = np.array(keys, dtype=np.uint64)
    block["v"] = np.arange(len(keys), dtype=np.int64)
    return block


def test_backend_has_a_stable_identity_token() -> None:
    ident = NumpyBackend().identity
    assert isinstance(ident, str) and ident


def test_partition_matches_the_golden_routing_vectors() -> None:
    # (a-golden) on numpy: the SAME frozen table the awkward backend must satisfy.
    be = NumpyBackend()
    for key, parts, dest in GOLDEN:
        subs = be.partition(_single_key_block(key), "__joinkey__", parts)
        assert len(subs) == parts
        assert len(subs[dest]) == 3, f"key {key:#x} into {parts} parts must fill dest {dest}"
        assert sum(len(s) for i, s in enumerate(subs) if i != dest) == 0


def test_partition_is_deterministic_and_conserves_rows() -> None:
    be = NumpyBackend()
    block = _block(list(range(64)))
    a = be.partition(block, "__joinkey__", 8)
    b = be.partition(block, "__joinkey__", 8)
    assert [len(s) for s in a] == [len(s) for s in b]  # (a) same routing on repeat
    assert sum(len(s) for s in a) == 64  # loses/duplicates no rows


def test_concat_is_in_order_and_total_length() -> None:
    be = NumpyBackend()
    merged = be.concat([_single_key_block(1, 2), _single_key_block(2, 3)])
    assert len(merged) == 5
    assert list(merged["__joinkey__"]) == [1, 1, 2, 2, 2]


def test_slice_rows_is_a_contiguous_record_slice() -> None:
    be = NumpyBackend()
    block = _block(list(range(10)))
    sl = be.slice_rows(block, 2, 6)
    assert len(sl) == 4
    assert list(sl["__joinkey__"]) == [2, 3, 4, 5]


def test_estimated_bytes_scales_with_itemsize() -> None:
    be = NumpyBackend()
    narrow = np.zeros(100, dtype=np.dtype([("k", np.uint8)]))
    wide = np.zeros(100, dtype=np.dtype([("k", np.uint64)]))
    assert be.estimated_bytes(wide) > be.estimated_bytes(narrow) > 0


def test_wire_roundtrip_preserves_record_columns() -> None:
    be = NumpyBackend()
    block = _block([10, 20, 30, 40, 50])
    back = be.from_wire(be.to_wire(block))
    assert list(back["__joinkey__"]) == list(block["__joinkey__"])
    assert list(back["v"]) == list(block["v"])


def test_wire_output_is_bytes() -> None:
    assert isinstance(NumpyBackend().to_wire(_single_key_block(5)), (bytes, bytearray))
