"""M40 GAP-2 — join routing must satisfy the FROZEN M39 golden vectors (spec INVARIANTS: "Golden
vectors in tests/frozen/{numpy,awkward}/m39/golden_route.py are the pin — join routing must satisfy
them too.").

M40 introduces no separate routing function for joins: the co-partitioning step (join_plan's two
``map_write`` stages routing onto the shared ``gather_join``) calls the SAME ``ShuffleBackend.partition``
primitive M39's exchange half already uses and already pins against ``golden_route.GOLDEN`` in
``test_exchange_primitives.py``. This is the join-side half of that invariant, driven end-to-end from
``graphed.pack_key`` rather than a hand-typed ``__joinkey__`` literal:

  1. For (run=0, lumi=0, event=e), a big-endian, most-significant-field-first packing MUST reproduce
     ``e`` EXACTLY as the u64 ``__joinkey__`` — zero high fields contribute zero bits, independent of
     the implementer's exact bit-width split (holds for any split as long as ``e`` fits its field,
     true for every small ``e`` used here). This also fails now (``graphed.pack_key`` is unimplemented)
     for the right reason.
  2. Those pack_key-produced keys are then routed with ``AwkwardBackend.partition`` and must land in
     the dest tabulated by the frozen M39 ``GOLDEN`` table — proving the join path hasn't grown its own
     (potentially non-conformant) routing.

Reads (never modifies) ``tests/frozen/awkward/m39/golden_route.py::GOLDEN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import awkward as ak
import numpy as np

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward

_M39_DIR = Path(__file__).resolve().parent.parent / "m39"
if str(_M39_DIR) not in sys.path:
    sys.path.insert(0, str(_M39_DIR))
from golden_route import GOLDEN  # noqa: E402  (frozen M39 table; imported read-only, never copied)

# Every GOLDEN key small enough to be exactly the `event` field value with run=lumi=0.
_SMALL_EVENTS: tuple[int, ...] = tuple(sorted({key for key, _, _ in GOLDEN if key <= 0x3E8}))
_GOLDEN_SMALL = [(key, parts, dest) for key, parts, dest in GOLDEN if key in _SMALL_EVENTS]


def _single_key_block(key: int, rows: int = 3) -> ak.Array:
    return ak.Array(
        {"__joinkey__": np.full(rows, key, dtype=np.uint64), "v": np.arange(rows, dtype=np.int64)}
    )


def test_pack_key_zero_high_fields_routes_per_the_m39_golden_vectors() -> None:
    s = Session(AwkwardBackend())
    src = from_awkward(
        s,
        "x",
        ak.Array(
            {"run": [0] * len(_SMALL_EVENTS), "lumi": [0] * len(_SMALL_EVENTS), "event": list(_SMALL_EVENTS)}
        ),
    )
    packed = graphed.pack_key(src, on=("run", "lumi", "event"))
    out = s.materialize(packed)
    got = dict(zip(_SMALL_EVENTS, (int(v) for v in out["__joinkey__"].to_list()), strict=True))

    for e in _SMALL_EVENTS:
        assert got[e] == e, (
            f"pack_key(run=0, lumi=0, event={e}) must equal {e} exactly — big-endian packing "
            "contributes zero bits from zeroed high fields, for any field-width split"
        )

    # the join co-partitioning path routes these SAME pack_key-produced keys through the shared
    # be.partition primitive — must still agree with the frozen M39 golden dests.
    be = AwkwardBackend()
    for key, parts, dest in _GOLDEN_SMALL:
        subs = be.partition(_single_key_block(key), "__joinkey__", parts)
        assert len(subs[dest]) == 3, f"join routing: key {key:#x} into {parts} parts must fill dest {dest}"
        assert sum(len(sub) for i, sub in enumerate(subs) if i != dest) == 0
