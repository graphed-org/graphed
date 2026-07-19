"""M40 GAP-1 — direct witness for ``graphed.pack_key`` (spec Implementation Target 8; trap #6).

See ``tests/frozen/awkward/m40/test_pack_key.py`` for the full rationale and the measured proof that
a ``hash()``-based packer violates (iii)/(iv)/(v) below; this file is the numpy-backend half of the
same direct witness (the existing ``test_join_primitives.py``/``test_op_form_join.py`` construct
``__joinkey__`` BY HAND on tiny int fixtures, so ``pack_key`` itself has no coverage there).

Pinned surface: ``graphed.pack_key(array, *, on)`` — a thin NEUTRAL frontend verb returning an
``Array`` carrying a flat u64 ``__joinkey__`` column, big-endian bit-packed most-significant field
(``run``) first. This file pins the STRUCTURE, not an exact width:

  (i)   determinism across a fresh subprocess with a different ``PYTHONHASHSEED``.
  (ii)  injectivity — N distinct triples -> N distinct ``__joinkey__``.
  (iii) low-field monotonicity — fixed (run, lumi), strictly increasing in ``event``.
  (iv)  high-field dominance — incrementing ``run`` outranks every key sharing the smaller ``run``.
  (v)   not ``hash()`` — ``__joinkey__`` != ``hash((run, lumi, event))`` and != hash of the field bytes.

(hash() fails iii/iv — measured with CPython 3.12 in the awkward-side docstring; int/tuple hashing is
unsalted so it does NOT fail (i), which instead guards other nondeterminism sources.)
"""

from __future__ import annotations

import itertools
import subprocess
import sys

import numpy as np

import graphed
from graphed import Session
from graphed.numpy import NumpyBackend, from_record

# Same 12 triples as the awkward-side witness (kept small so any reasonable bit-width allocation
# the implementer picks still satisfies the monotonicity/dominance assertions).
ROWS: list[tuple[int, int, int]] = [
    (1, 0, 0),
    (1, 0, 1),
    (1, 0, 2),
    (1, 0, 50),
    (1, 0, 500),
    (1, 7, 500),
    (1, 63, 500),
    (2, 0, 0),
    (2, 0, 500),
    (2, 7, 500),
    (2, 63, 500),
    (3, 0, 0),
]

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _packed(rows: list[tuple[int, int, int]]) -> list[int]:
    s = Session(NumpyBackend())
    src = from_record(
        s,
        "x",
        run=np.array([r for r, _, _ in rows], dtype=np.int64),
        lumi=np.array([lu for _, lu, _ in rows], dtype=np.int64),
        event=np.array([e for _, _, e in rows], dtype=np.int64),
    )
    packed = graphed.pack_key(src, on=("run", "lumi", "event"))
    out = s.materialize(packed)
    return [int(v) for v in np.asarray(out["__joinkey__"])]


def test_pack_key_is_injective_on_distinct_triples() -> None:
    # (ii) kills a packer that collapses fields (e.g. keys only on `event`).
    vals = _packed(ROWS)
    assert len(vals) == len(ROWS)
    assert len(set(vals)) == len(ROWS), "distinct (run,lumi,event) triples must pack to distinct keys"


def test_pack_key_is_strictly_monotonic_in_event_for_fixed_run_lumi() -> None:
    # (iii) fixed (run=1, lumi=0), events ascending -> __joinkey__ strictly increasing.
    mono_rows = [row for row in ROWS if row[0] == 1 and row[1] == 0]
    assert [e for _, _, e in mono_rows] == [0, 1, 2, 50, 500]
    vals = _packed(mono_rows)
    assert all(a < b for a, b in itertools.pairwise(vals)), (
        "__joinkey__ must be strictly increasing in event for fixed (run, lumi)"
    )


def test_pack_key_run_dominates_lumi_and_event() -> None:
    # (iv) big-endian dominance: every run=2 key must exceed every run=1 key over the tested range.
    vals = dict(zip(ROWS, _packed(ROWS), strict=True))
    run1 = [v for row, v in vals.items() if row[0] == 1]
    run2 = [v for row, v in vals.items() if row[0] == 2]
    assert max(run1) < min(run2), "run must dominate: any run=2 key > every run=1 key (big-endian)"


def test_pack_key_is_not_pythons_hash() -> None:
    # (v) direct inequality with hash() — kills `pack_key = lambda t: hash(t)` outright.
    vals = dict(zip(ROWS, _packed(ROWS), strict=True))
    for (r, lu, e), packed in vals.items():
        assert packed != (hash((r, lu, e)) & _MASK64), f"{(r, lu, e)}: __joinkey__ must not be hash(triple)"
        field_bytes = r.to_bytes(8, "big") + lu.to_bytes(8, "big") + e.to_bytes(8, "big")
        assert packed != (hash(field_bytes) & _MASK64), f"{(r, lu, e)}: __joinkey__ must not be hash(bytes)"


# (i) cross-process / cross-hash-seed determinism — mirrors graphed-exec-check's M39 B2 pattern
# (tests/frozen/m39/test_routing_invariance.py), adapted to pack_key.
_CHILD = r"""
import sys
import numpy as np
import graphed
from graphed import Session
from graphed.numpy import NumpyBackend, from_record
triples = [tuple(int(x) for x in tok.split(",")) for tok in sys.argv[1:]]
s = Session(NumpyBackend())
src = from_record(
    s, "x",
    run=np.array([t[0] for t in triples], dtype=np.int64),
    lumi=np.array([t[1] for t in triples], dtype=np.int64),
    event=np.array([t[2] for t in triples], dtype=np.int64),
)
out = s.materialize(graphed.pack_key(src, on=("run", "lumi", "event")))
print(" ".join(str(int(v)) for v in np.asarray(out["__joinkey__"])))
"""


def _pack_in_child(seed: str) -> list[str]:
    args = [f"{r},{lu},{e}" for r, lu, e in ROWS]
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, *args],
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()


def test_pack_key_is_stable_across_processes_and_hash_seeds() -> None:
    a = _pack_in_child("0")
    b = _pack_in_child("1")
    assert a == b, "pack_key must be independent of PYTHONHASHSEED (never Python hash() in the path)"
    assert a == [str(v) for v in _packed(ROWS)], "and match the in-process computation"
