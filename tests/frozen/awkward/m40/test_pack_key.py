"""M40 GAP-1 — direct witness for ``graphed.pack_key`` (spec Implementation Target 8; trap #6).

Every other M40 primitives test builds its ``__joinkey__`` column BY HAND (see
``test_join_primitives.py``'s ``_block(__joinkey__=[1, 2, ...])``), so a FORBIDDEN
``pack_key = lambda row: hash(row)`` implementation is never exercised anywhere in this suite — and
on the tiny fixture keys used elsewhere (small ints, no collisions, no ordering asserted across
fields) it would silently pass every other test. This file is the missing direct witness.

Pinned surface (this contract; the implementer must build it): ``graphed.pack_key(array, *, on)`` — a
thin NEUTRAL frontend verb (mirrors ``graphed.join``) returning an ``Array`` carrying a flat unsigned-
64 ``__joinkey__`` column, built by BIG-ENDIAN integer bit-ops over the key fields, most-significant
field first (plan §2.1/§3.3: ``run`` … ``event``). ``graphed.join`` calls it internally. Exact field
bit-widths are the implementer's choice (spec: "impl's choice"), so this file pins the STRUCTURE a
big-endian integer packing has and ``hash()`` lacks, not a specific numeric value:

  (i)   DETERMINISM — same triple -> same u64, unchanged when recomputed in a FRESH SUBPROCESS
        launched with a different ``PYTHONHASHSEED`` (mirrors the M39 B2 pattern in
        ``graphed-exec-check/tests/frozen/m39/test_routing_invariance.py``).
  (ii)  INJECTIVITY — N distinct (run, lumi, event) triples -> N distinct ``__joinkey__`` values.
  (iii) LOW-FIELD MONOTONICITY — for a fixed (run, lumi), ``__joinkey__`` is STRICTLY increasing in
        ``event``.
  (iv)  HIGH-FIELD DOMINANCE (big-endian) — incrementing ``run`` yields a LARGER ``__joinkey__`` than
        ANY key sharing the smaller ``run`` (any lumi/event in the tested range) — ``run`` dominates.
  (v)   NOT ``hash()`` — ``__joinkey__`` != Python ``hash((run, lumi, event))`` and != a hash of the
        fields' big-endian bytes, for every tested triple.

MEASURED PROOF that (iii)/(iv)/(v) actually discriminate a ``hash()``-based packer (run at authoring
time, CPython 3.12, this repo's venv; reproduced here as a comment, not committed as code)::

    >>> def _hash_packkey(r, l, e):
    ...     return hash((r, l, e)) & 0xFFFFFFFFFFFFFFFF
    >>> seq = [_hash_packkey(1, 0, e) for e in (0, 1, 2, 50, 500)]
    >>> all(a < b for a, b in zip(seq, seq[1:]))   # (iii) monotonicity
    False   # measured: [7839147235533974352, 2946073206561277986, 9453713962663276411, ...] — NOT sorted
    >>> run1 = [_hash_packkey(1, l, e) for l in (0, 7, 63) for e in (0, 1, 2, 50, 500)]
    >>> run2 = [_hash_packkey(2, l, e) for l in (0, 7, 63) for e in (0, 1, 2, 50, 500)]
    >>> max(run1) < min(run2)   # (iv) dominance
    False   # measured: max(run1)=17689295887498362404, min(run2)=114010025686004027

(v) holds by construction (we compare directly against ``hash()``). Note (i) does NOT discriminate
``hash()`` here: CPython's ``int``/tuple-of-``int`` hash is unsalted (only str/bytes are
``PYTHONHASHSEED``-randomized), so ``hash((run, lumi, event))`` is already process-invariant — (i)
guards against OTHER nondeterminism (dict/set iteration order, id()-based folding), not this trap.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys

import awkward as ak

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward

# 12 distinct (run, lumi, event) triples: a monotonicity ladder at (run=1, lumi=0), extra lumi values
# at run=1, and run=2/run=3 rows for the dominance check. All field values stay small (<= 500) so the
# assertions hold under ANY reasonable big-endian bit-width allocation the implementer picks.
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
    s = Session(AwkwardBackend())
    src = from_awkward(
        s,
        "x",
        ak.Array(
            {
                "run": [r for r, _, _ in rows],
                "lumi": [lu for _, lu, _ in rows],
                "event": [e for _, _, e in rows],
            }
        ),
    )
    packed = graphed.pack_key(src, on=("run", "lumi", "event"))
    out = s.materialize(packed)
    return [int(v) for v in out["__joinkey__"].to_list()]


def test_pack_key_is_injective_on_distinct_triples() -> None:
    # (ii) 12 distinct triples must yield 12 distinct u64s. Kills a packer that collapses fields
    # (e.g. keys only on `event`, ignoring run/lumi).
    vals = _packed(ROWS)
    assert len(vals) == len(ROWS)
    assert len(set(vals)) == len(ROWS), "distinct (run,lumi,event) triples must pack to distinct keys"


def test_pack_key_is_strictly_monotonic_in_event_for_fixed_run_lumi() -> None:
    # (iii) fixed (run=1, lumi=0), events [0,1,2,50,500] ascending -> __joinkey__ strictly increasing.
    # A hash()-based packer is NOT sorted here (measured in the module docstring) -> FAILS.
    mono_rows = [row for row in ROWS if row[0] == 1 and row[1] == 0]
    assert [e for _, _, e in mono_rows] == [0, 1, 2, 50, 500]
    vals = _packed(mono_rows)
    assert all(a < b for a, b in itertools.pairwise(vals)), (
        "__joinkey__ must be strictly increasing in event for fixed (run, lumi) — big-endian, "
        "event least-significant"
    )


def test_pack_key_run_dominates_lumi_and_event() -> None:
    # (iv) big-endian dominance: every run=2 key must exceed every run=1 key over the tested lumi/event
    # range. A hash()-based packer violates this (measured in the module docstring) -> FAILS.
    vals = dict(zip(ROWS, _packed(ROWS), strict=True))
    run1 = [v for row, v in vals.items() if row[0] == 1]
    run2 = [v for row, v in vals.items() if row[0] == 2]
    assert max(run1) < min(run2), "run must dominate: any run=2 key > every run=1 key (big-endian)"


def test_pack_key_is_not_pythons_hash() -> None:
    # (v) direct inequality with hash() — kills `pack_key = lambda t: hash(t)` (and a hash-of-bytes
    # variant) outright, independent of the structural checks above.
    vals = dict(zip(ROWS, _packed(ROWS), strict=True))
    for (r, lu, e), packed in vals.items():
        assert packed != (hash((r, lu, e)) & _MASK64), f"{(r, lu, e)}: __joinkey__ must not be hash(triple)"
        field_bytes = r.to_bytes(8, "big") + lu.to_bytes(8, "big") + e.to_bytes(8, "big")
        assert packed != (hash(field_bytes) & _MASK64), f"{(r, lu, e)}: __joinkey__ must not be hash(bytes)"


# (i) cross-process / cross-hash-seed determinism — mirrors the M39 B2 pattern in
# graphed-exec-check/tests/frozen/m39/test_routing_invariance.py, adapted to pack_key.
_CHILD = r"""
import sys
import awkward as ak
import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
triples = [tuple(int(x) for x in tok.split(",")) for tok in sys.argv[1:]]
s = Session(AwkwardBackend())
src = from_awkward(
    s, "x",
    ak.Array({"run": [t[0] for t in triples], "lumi": [t[1] for t in triples], "event": [t[2] for t in triples]}),
)
out = s.materialize(graphed.pack_key(src, on=("run", "lumi", "event")))
print(" ".join(str(int(v)) for v in out["__joinkey__"].to_list()))
"""


def _pack_in_child(seed: str) -> list[str]:
    args = [f"{r},{lu},{e}" for r, lu, e in ROWS]
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, *args],
        # Inherit the parent environment (so the child can start on every OS — a hardcoded POSIX
        # `PATH=/usr/bin:/bin` with no SystemRoot makes the Windows child fail Winsock init when
        # `import awkward` pulls in asyncio, WinError 10106) and override ONLY PYTHONHASHSEED — the
        # single variable under test. The two children still differ in that one variable, so a
        # PYTHONHASHSEED-sensitive packer (dict/set-ordering, id-folding) is still caught.
        env={**os.environ, "PYTHONHASHSEED": seed},
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
