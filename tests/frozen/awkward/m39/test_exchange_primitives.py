"""M39 — the awkward ``ShuffleBackend`` exchange primitives (plan §3.0, §3.3).

``AwkwardBackend`` supplies the vectorized exchange half the generic engine calls:
``partition``/``concat``/``slice_rows``/``estimated_bytes``/``to_wire``/``from_wire`` (+ ``identity``).
The load-bearing witness is (a-golden): ``partition`` MUST reproduce the pinned §4 SHA-256 dest
assignment recorded in ``golden_route.GOLDEN`` — a backend substituting any other process-independent
hash (blake2b/xxhash) FAILS these vectors even though it would pass the B2 process-invariance check.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from golden_route import GOLDEN

from graphed.awkward import AwkwardBackend


def _single_key_block(key: int, rows: int = 3) -> ak.Array:
    """A record block whose every row carries the same packed-u64 ``__joinkey__``."""
    return ak.Array(
        {"__joinkey__": np.full(rows, key, dtype=np.uint64), "v": np.arange(rows, dtype=np.int64)}
    )


def _jagged_block(keys: list[int]) -> ak.Array:
    return ak.Array(
        {
            "__joinkey__": np.array(keys, dtype=np.uint64),
            "jets": ak.Array([list(range(i % 4)) for i in range(len(keys))]),
        }
    )


def test_backend_has_a_stable_identity_token() -> None:
    ident = AwkwardBackend().identity
    assert isinstance(ident, str) and ident, "identity is the versioned shuffle-format token (§7.2)"


def test_partition_matches_the_golden_routing_vectors() -> None:
    # (a-golden) THE conformance witness: every single-key block lands entirely in the tabulated dest.
    be = AwkwardBackend()
    for key, parts, dest in GOLDEN:
        subs = be.partition(_single_key_block(key), "__joinkey__", parts)
        assert len(subs) == parts
        assert len(subs[dest]) == 3, f"key {key:#x} into {parts} parts must fill dest {dest}"
        assert sum(len(s) for i, s in enumerate(subs) if i != dest) == 0, "no rows in any other dest"


def test_partition_is_deterministic_across_calls() -> None:
    # (a) same key -> same dest, witnessed by identical per-row routing on repeat.
    be = AwkwardBackend()
    block = _jagged_block([1, 2, 3, 7, 42, 255, 1, 2, 3])
    a = [len(s) for s in be.partition(block, "__joinkey__", 8)]
    b = [len(s) for s in be.partition(block, "__joinkey__", 8)]
    assert a == b


def test_partition_conserves_every_row() -> None:
    be = AwkwardBackend()
    block = _jagged_block(list(range(50)))
    subs = be.partition(block, "__joinkey__", 8)
    assert sum(len(s) for s in subs) == 50, "route+split must lose or duplicate no rows"


def test_concat_is_in_order_and_total_length() -> None:
    be = AwkwardBackend()
    a = _single_key_block(1, rows=2)
    b = _single_key_block(2, rows=3)
    merged = be.concat([a, b])
    assert len(merged) == 5
    assert merged["__joinkey__"].to_list() == [1, 1, 2, 2, 2], "vertical concat preserves order"


def test_slice_rows_preserves_jagged_structure() -> None:
    # the §5.1 jagged split primitive: an event-boundary slice keeps each row's list intact.
    be = AwkwardBackend()
    block = _jagged_block(list(range(10)))
    sl = be.slice_rows(block, 2, 6)
    assert len(sl) == 4
    assert sl["jets"].to_list() == block["jets"][2:6].to_list()


def test_estimated_bytes_tracks_payload_size_not_row_count() -> None:
    # jagged bytes != entry count (§5.1): same #rows, bigger lists -> more bytes.
    be = AwkwardBackend()
    small = ak.Array({"jets": ak.Array([[0]] * 20)})
    big = ak.Array({"jets": ak.Array([list(range(50))] * 20)})
    assert be.estimated_bytes(big) > be.estimated_bytes(small)
    assert be.estimated_bytes(small) > 0


def test_wire_roundtrip_preserves_values_and_jaggedness() -> None:
    be = AwkwardBackend()
    block = _jagged_block([10, 20, 30, 40, 50])
    back = be.from_wire(be.to_wire(block))
    assert back["__joinkey__"].to_list() == block["__joinkey__"].to_list()
    assert back["jets"].to_list() == block["jets"].to_list()


def test_wire_output_is_bytes() -> None:
    assert isinstance(AwkwardBackend().to_wire(_single_key_block(5)), (bytes, bytearray))
