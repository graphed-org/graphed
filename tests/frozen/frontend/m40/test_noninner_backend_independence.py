"""M40 iter-2 — NON-INNER ``graphed.join`` (left/outer) is backend-independent AND relationally
correct against a NULL-preserving, KEY-COALESCING pandas oracle (contract F3/F4; plan §3.3).

The iteration-1 frozen suite (``test_relational_join.py``) only pins ``how="inner"``, so every
non-inner code path — unmatched-row survival, absent-side nulls, and (the sharp one) SQL COALESCE of
the ``on`` key columns on an orphan row — is UNCOVERED. This file extends the M2 two-backend
discipline (``shuffle_backends.REAL_BACKENDS`` = {AwkwardBackend, NumpyBackend}) to ``how`` ∈
{left, outer}: the SAME recorded ``graphed.join`` is materialized on each backend and normalized to
the SAME canonical relational rows, then asserted bit-for-bit against ``pandas.merge(how=...)``.

The oracle is DUPLICATING (k*m rows per key), NULL-preserving (an absent side's non-key field reads
NULL, represented as the ``None`` sentinel — never coerced to an int), and KEY-COALESCING (pandas
never nulls a key column on a miss; the surviving side supplies it). The normalizer treats the
awkward option ``None`` and the numpy masked element identically, so a backend that leaks its own
null spelling still compares equal only when the RELATIONAL value matches.

Why these FAIL against the current impl (measured, authoring venv, commit 4bc452e):
``graphed.join`` records only the packed ``{"on": "__joinkey__"}`` — the real ``on`` fields never
reach ``merge_records``, so on a build-absent orphan (right-only key ``event=99``) BOTH backends
return ``run=lumi=event=None`` instead of the coalesced ``(1, 1, 99)``::

    how=outer  oracle (1, 1, 99, None, 0)  |  awkward got (None, None, None, None, 0)
    how=outer  oracle (1, 1, 99, None, 0)  |  numpy   got (None, None, None, None, 0)

``how="left"`` already matches on both (no build-absent orphan, so no coalescing is exercised) — it
is asserted here anyway as the backend-independence / regression pin; the combined per-backend test
FAILS on its ``outer`` leg.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
import pytest
from shuffle_backends import COLUMNS, ON, REAL_BACKENDS, BackendCase, skim_tables

import graphed
from graphed import Session

_IDS = [c.name for c in REAL_BACKENDS]
_KEYIDX = tuple(COLUMNS.index(k) for k in ON)


def _cell(v: Any) -> int | None:
    """A single relational cell, null-normalized: awkward option ``None``, a numpy masked element, or
    a pandas float ``NaN`` all collapse to the ``None`` sentinel; everything else is an ``int``."""
    if v is None or v is np.ma.masked or np.ma.is_masked(v):
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return int(v)


def _srt(rows: list[tuple[int | None, ...]]) -> list[tuple[int | None, ...]]:
    # None-safe multiset sort (can't order None against int directly).
    return sorted(rows, key=lambda r: tuple((x is None, -1 if x is None else x) for x in r))


def _rows(name: str, block: Any) -> list[tuple[int | None, ...]]:
    """Materialized join block -> sorted multiset of relational rows over ``COLUMNS``, nulls kept.

    The numpy block is a column ``Mapping`` OR a (possibly masked) structured array; both are indexed
    field-then-row DIRECTLY (``block[c][i]``), never via ``np.asarray`` — that would strip a
    ``MaskedArray``'s mask and resurrect the gathered ``take(-1)`` leftover, hiding a genuine null."""
    if name == "awkward":
        return _srt([tuple(_cell(r[c]) for c in COLUMNS) for r in ak.to_list(block)])
    n = len(block[COLUMNS[0]])
    return _srt([tuple(_cell(block[c][i]) for c in COLUMNS) for i in range(n)])


def _oracle(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], *, how: str
) -> list[tuple[int | None, ...]]:
    """The DUPLICATING, NULL-preserving, KEY-COALESCING reference: ``pandas.merge(how=...)``. A
    missing non-key cell is ``NaN`` -> ``None``; the ``on`` keys are coalesced by pandas (never null)."""
    ref = pd.merge(pd.DataFrame(dict(left)), pd.DataFrame(dict(right)), on=list(ON), how=how)
    return _srt(
        [tuple(None if pd.isna(ref[c].iloc[i]) else int(ref[c].iloc[i]) for c in COLUMNS) for i in range(len(ref))]
    )


def _join_rows(case: BackendCase, *, how: str) -> list[tuple[int | None, ...]]:
    left_cols, right_cols = skim_tables()
    s = Session(case.make_backend())
    left = case.make_source(s, "left", left_cols)
    right = case.make_source(s, "right", right_cols)
    return _rows(case.name, s.materialize(graphed.join(left, right, on=ON, how=how)))


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_noninner_join_equals_null_preserving_key_coalescing_pandas(case: BackendCase) -> None:
    # F3/F4: the SAME left+outer suite must be bit-for-bit the pandas oracle on BOTH backends. Fails
    # now on the `outer` leg: the right-only orphan key (event=99) comes back run=lumi=event=None
    # instead of the coalesced (1, 1, 99).
    left_cols, right_cols = skim_tables()
    for how in ("left", "outer"):
        got = _join_rows(case, how=how)
        expected = _oracle(left_cols, right_cols, how=how)
        assert got == expected, (
            f"{case.name}: how={how} must equal the null-preserving key-coalescing pandas merge "
            f"(relational multiset + null-ness)\n  got={got}\n  exp={expected}"
        )


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_outer_orphan_key_columns_are_coalesced_not_null(case: BackendCase) -> None:
    # F3 sharp witness: on a build-absent (right-only) orphan the user's `on` key columns take the
    # PRESENT side's value (SQL COALESCE) — NEVER null. Only the absent side's NON-key field (njet) is
    # null. Fails now: current impl nulls the whole key triple on the orphan.
    rows = _join_rows(case, how="outer")
    orphans = [r for r in rows if r[COLUMNS.index("event")] == 99]
    if not orphans:  # the orphan key is dropped/nulled -> it can't be found by event=99 at all
        keys_seen = sorted({r[COLUMNS.index("event")] for r in rows}, key=lambda e: (e is None, e or -1))
        raise AssertionError(
            f"{case.name}: right-only orphan event=99 absent from outer join (key not coalesced); "
            f"events seen={keys_seen}"
        )
    (row,) = orphans
    for k in ON:
        assert row[COLUMNS.index(k)] is not None, (
            f"{case.name}: outer orphan key column {k!r} is null — must be coalesced from the present side"
        )
    assert row == (1, 1, 99, None, 0), f"{case.name}: outer orphan row must coalesce keys, null only njet: {row}"
