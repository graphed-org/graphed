"""FROZEN golden routing vectors for the M39 shuffle hash-route conformance theme (a-golden).

Every ``ShuffleBackend.partition`` (awkward AND numpy) MUST place a single-key block into exactly the
dest tabulated here. The vectors are PRECOMPUTED (not derived in-test) from the pinned rule of plan
§4/§3.0::

    route(key, P) = int.from_bytes(sha256(key.to_bytes(8, "big")).digest()[:8], "big") % P

with ``salt=0`` (no bytes appended). They were generated with CPython ``hashlib`` and committed as
literals so the routing rule cannot be changed and the test updated in tandem — the table is the pin.

DISCRIMINATION (measured at authoring time): of these 96 vectors, sha256 and blake2b-of-the-same-key
disagree on 81. A backend that substitutes ANY other process-independent hash (blake2b, xxhash)
passes the B2 cross-process-invariance witness yet FAILS these vectors — which is the whole point
(ADV-r5.3: ``@runtime_checkable`` cannot witness the routing rule; these vectors do).

``key`` is the packed u64 ``__joinkey__`` value; ``canonical_be`` is its 8-byte big-endian encoding.
"""

from __future__ import annotations

import hashlib

# (packed-u64 key, parts P, expected dest sub-block index)
GOLDEN: list[tuple[int, int, int]] = [
    (0x0000000000000000, 2, 0),
    (0x0000000000000000, 3, 2),
    (0x0000000000000000, 4, 2),
    (0x0000000000000000, 8, 2),
    (0x0000000000000000, 16, 10),
    (0x0000000000000000, 1000, 850),
    (0x0000000000000001, 2, 0),
    (0x0000000000000001, 3, 2),
    (0x0000000000000001, 4, 2),
    (0x0000000000000001, 8, 2),
    (0x0000000000000001, 16, 2),
    (0x0000000000000001, 1000, 730),
    (0x0000000000000002, 2, 1),
    (0x0000000000000002, 3, 2),
    (0x0000000000000002, 4, 1),
    (0x0000000000000002, 8, 5),
    (0x0000000000000002, 16, 13),
    (0x0000000000000002, 1000, 13),
    (0x0000000000000003, 2, 0),
    (0x0000000000000003, 3, 1),
    (0x0000000000000003, 4, 0),
    (0x0000000000000003, 8, 4),
    (0x0000000000000003, 16, 12),
    (0x0000000000000003, 1000, 948),
    (0x0000000000000007, 2, 0),
    (0x0000000000000007, 3, 2),
    (0x0000000000000007, 4, 0),
    (0x0000000000000007, 8, 4),
    (0x0000000000000007, 16, 12),
    (0x0000000000000007, 1000, 212),
    (0x000000000000002A, 2, 0),
    (0x000000000000002A, 3, 2),
    (0x000000000000002A, 4, 2),
    (0x000000000000002A, 8, 2),
    (0x000000000000002A, 16, 10),
    (0x000000000000002A, 1000, 938),
    (0x00000000000000FF, 2, 0),
    (0x00000000000000FF, 3, 2),
    (0x00000000000000FF, 4, 2),
    (0x00000000000000FF, 8, 2),
    (0x00000000000000FF, 16, 10),
    (0x00000000000000FF, 1000, 490),
    (0x0000000000000100, 2, 1),
    (0x0000000000000100, 3, 2),
    (0x0000000000000100, 4, 3),
    (0x0000000000000100, 8, 7),
    (0x0000000000000100, 16, 15),
    (0x0000000000000100, 1000, 983),
    (0x00000000000003E8, 2, 0),
    (0x00000000000003E8, 3, 1),
    (0x00000000000003E8, 4, 0),
    (0x00000000000003E8, 8, 4),
    (0x00000000000003E8, 16, 4),
    (0x00000000000003E8, 1000, 516),
    (0x0000000100000000, 2, 1),
    (0x0000000100000000, 3, 0),
    (0x0000000100000000, 4, 3),
    (0x0000000100000000, 8, 3),
    (0x0000000100000000, 16, 3),
    (0x0000000100000000, 1000, 795),
    (0x8000000000000000, 2, 1),
    (0x8000000000000000, 3, 2),
    (0x8000000000000000, 4, 1),
    (0x8000000000000000, 8, 5),
    (0x8000000000000000, 16, 5),
    (0x8000000000000000, 1000, 437),
    (0xFFFFFFFFFFFFFFFF, 2, 1),
    (0xFFFFFFFFFFFFFFFF, 3, 2),
    (0xFFFFFFFFFFFFFFFF, 4, 1),
    (0xFFFFFFFFFFFFFFFF, 8, 5),
    (0xFFFFFFFFFFFFFFFF, 16, 13),
    (0xFFFFFFFFFFFFFFFF, 1000, 325),
    (0x00000000075BCD15, 2, 1),
    (0x00000000075BCD15, 3, 0),
    (0x00000000075BCD15, 4, 3),
    (0x00000000075BCD15, 8, 3),
    (0x00000000075BCD15, 16, 11),
    (0x00000000075BCD15, 1000, 35),
    (0x000000003ADE68B1, 2, 1),
    (0x000000003ADE68B1, 3, 1),
    (0x000000003ADE68B1, 4, 1),
    (0x000000003ADE68B1, 8, 5),
    (0x000000003ADE68B1, 16, 5),
    (0x000000003ADE68B1, 1000, 773),
    (0x00000000DEADBEEF, 2, 0),
    (0x00000000DEADBEEF, 3, 1),
    (0x00000000DEADBEEF, 4, 0),
    (0x00000000DEADBEEF, 8, 4),
    (0x00000000DEADBEEF, 16, 4),
    (0x00000000DEADBEEF, 1000, 516),
    (0x00000000CAFEBABE, 2, 0),
    (0x00000000CAFEBABE, 3, 1),
    (0x00000000CAFEBABE, 4, 0),
    (0x00000000CAFEBABE, 8, 0),
    (0x00000000CAFEBABE, 16, 0),
    (0x00000000CAFEBABE, 1000, 336),
]


def canonical_be(key: int) -> bytes:
    """The pinned canonical big-endian encoding of a packed-u64 join key (plan §4)."""
    return int(key).to_bytes(8, "big")


def reference_route(key: int, parts: int) -> int:
    """The pinned §4 routing rule, recomputed — used ONLY to prove the frozen table is self-consistent,
    never as the conformance oracle (the frozen literals above are the oracle)."""
    return int.from_bytes(hashlib.sha256(canonical_be(key)).digest()[:8], "big") % parts
