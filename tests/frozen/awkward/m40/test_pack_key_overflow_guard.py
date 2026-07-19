"""M40 iter-2 — ``graphed.pack_key`` OVERFLOW GUARD on the awkward backend (contract F8).

The current packer lays out ``run<<40 | lumi<<20 | event`` with 20-bit fields, then masks each field
to its width. A field value ``>= 2**20`` therefore does NOT error — it wraps, silently COLLIDING with
a different triple. Measured (authoring venv, commit 4bc452e)::

    pack(run=1, lumi=1, event=2**20) == pack(run=1, lumi=1, event=0) == 1099512676352   # SILENT COLLIDE

A silent key collision is a correctness hole: two distinct events hash to one join key and cross-
contaminate. This file pins the GUARD only — a field at/above the packing width must raise a LOUD
``ValueError`` — NOT a wider key space (the packing CEILING itself stays Phase-2). The boundary is
discriminating on both sides: ``event = 2**20 - 1`` (the largest value the 20-bit field holds) must
still pack cleanly; ``event = 2**20`` must raise.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward

_WIDTH = 2**20  # the per-field packing width the guard must defend (event field, plan §2.1/§3.3)


def _pack(events: list[int]) -> Any:
    s = Session(AwkwardBackend())
    src = from_awkward(s, "x", ak.Array({"run": [1] * len(events), "lumi": [1] * len(events), "event": events}))
    return s.materialize(graphed.pack_key(src, on=("run", "lumi", "event")))


def test_pack_key_raises_on_event_field_at_packing_width() -> None:
    # F8: event = 2**20 overflows the 20-bit field -> must raise ValueError (currently masks to 0 and
    # silently collides with event=0, no raise). Discriminating boundary: 2**20-1 packs cleanly and
    # does NOT collide with 0, so a ValueError from the 2**20 case can only be the overflow guard.
    ok = _pack([_WIDTH - 1, 0])
    vals = [int(v) for v in ok["__joinkey__"].to_list()]
    assert vals[0] != vals[1], "sanity: event=2**20-1 fits the field and must not collide with event=0"

    with pytest.raises(ValueError):
        _pack([_WIDTH, 0])
