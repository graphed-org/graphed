"""M40 iter-2 — ``graphed.pack_key`` OVERFLOW GUARD on the numpy backend (contract F8).

The numpy-backend half of the awkward witness (see
``tests/frozen/awkward/m40/test_pack_key_overflow_guard.py`` for the full rationale + the measured
collision). The packer lays out ``run<<40 | lumi<<20 | event`` with 20-bit fields and masks each
field to width, so a value ``>= 2**20`` wraps and silently collides. Measured (commit 4bc452e)::

    pack(run=1, lumi=1, event=2**20) == pack(run=1, lumi=1, event=0) == 1099512676352   # SILENT COLLIDE

Pins the GUARD only (raise a LOUD ``ValueError`` at/above the field width) — not a wider key space,
which stays Phase-2. Boundary-discriminating: ``event = 2**20 - 1`` packs cleanly, ``2**20`` raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.numpy import NumpyBackend, from_record

_WIDTH = 2**20  # the per-field packing width the guard must defend (event field, plan §2.1/§3.3)


def _pack(events: list[int]) -> Any:
    s = Session(NumpyBackend())
    n = len(events)
    src = from_record(
        s,
        "x",
        run=np.ones(n, dtype=np.int64),
        lumi=np.ones(n, dtype=np.int64),
        event=np.array(events, dtype=np.int64),
    )
    return s.materialize(graphed.pack_key(src, on=("run", "lumi", "event")))


def test_pack_key_raises_on_event_field_at_packing_width() -> None:
    # F8: event = 2**20 overflows the 20-bit field -> must raise ValueError (currently masks to 0 and
    # silently collides with event=0, no raise). Discriminating boundary: 2**20-1 packs cleanly and
    # does NOT collide with 0, so a ValueError from the 2**20 case can only be the overflow guard.
    ok = _pack([_WIDTH - 1, 0])
    vals = [int(v) for v in np.asarray(ok["__joinkey__"])]
    assert vals[0] != vals[1], "sanity: event=2**20-1 fits the field and must not collide with event=0"

    with pytest.raises(ValueError):
        _pack([_WIDTH, 0])
