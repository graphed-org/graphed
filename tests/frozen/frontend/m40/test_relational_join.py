"""M40 theme (a)/(a2) — ``graphed.join`` is a bit-for-bit relational (SQL-duplicating) inner join,
proven over BOTH shipping backends (plan §3.1, §3.3; contract FROZEN-TEST rows (a)/(a2)).

Every test parametrizes over ``REAL_BACKENDS`` = {AwkwardBackend, NumpyBackend}: the SAME recorded
program is materialized on each and normalized to the same canonical relational rows. This is the M2
two-backend discipline applied to the join half of the ``ShuffleBackend`` seam — a primitive that
leaked awkward-only semantics (jagged list-of-matches, native option) would fail on numpy.

The reference is a DUPLICATING pandas ``merge`` (``shuffle_backends.pandas_reference``), NOT a
list-of-matches grouping (contract trap 3): a key with k left and m right matches must produce k*m
output rows.

Pre-implementation these fail for the right reason: ``graphed`` exposes no ``join`` attribute
(``AttributeError``), so no result is ever produced.
"""

from __future__ import annotations

import pytest
from shuffle_backends import COLUMNS, ON, REAL_BACKENDS, BackendCase, pandas_reference, skim_tables

import graphed
from graphed import Session

_IDS = [c.name for c in REAL_BACKENDS]


def _join_rows(case: BackendCase, *, how: str = "inner") -> list[tuple[int, ...]]:
    """Record ``graphed.join`` over two skim-derived sources on ``case``'s backend, materialize it,
    and normalize the result to a sorted list of integer relational rows."""
    left_cols, right_cols = skim_tables()
    s = Session(case.make_backend())
    left = case.make_source(s, "left", left_cols)
    right = case.make_source(s, "right", right_cols)
    joined = graphed.join(left, right, on=ON, how=how)
    return case.to_rows(s.materialize(joined))


def test_graphed_join_is_a_neutral_module_verb() -> None:
    # a join is NEITHER an awkward nor a numpy idiom -> a module function, not an Array method
    # (mirror of the M39 repartition neutral-verb witness). Fails an impl that hangs join off Array.
    assert callable(graphed.join)
    assert not hasattr(graphed.Array, "join"), "join must be a module verb, not an Array method"


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_inner_join_is_bit_for_bit_the_duplicating_pandas_merge(case: BackendCase) -> None:
    left_cols, right_cols = skim_tables()
    expected = pandas_reference(left_cols, right_cols, how="inner")
    got = _join_rows(case, how="inner")
    assert got == expected, (
        f"{case.name}: join must equal the duplicating SQL merge (row multiset) bit-for-bit"
    )


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_join_duplicates_matches_not_list_of_matches(case: BackendCase) -> None:
    # THE relational-vs-grouped discriminator (contract trap 3). With key multiplicity 3x2 / 2x1 /
    # 1x2, a duplicating merge yields 6+2+2 = 10 rows across 3 distinct joined keys; a list-of-matches
    # (grouped) impl yields one nested row per matched probe (<= 6) and FAILS both assertions.
    got = _join_rows(case, how="inner")
    distinct_keys = {row[: len(ON)] for row in got}
    assert len(got) == 10, f"{case.name}: expected 10 duplicated rows, got {len(got)}"
    assert len(got) > len(distinct_keys), (
        f"{case.name}: duplication not exercised — {len(got)} rows for {len(distinct_keys)} keys"
    )


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_inner_join_drops_unmatched_orphan_keys(case: BackendCase) -> None:
    # how="inner": the left-only key event=88 and the right-only key event=99 must NOT appear. A
    # left/outer impl (or one that ignores `how`) keeps them (with nulls) and fails the row count.
    got = _join_rows(case, how="inner")
    events = {row[COLUMNS.index("event")] for row in got}
    assert 88 not in events and 99 not in events, f"{case.name}: inner join leaked an orphan key"
    assert events == {10, 20, 30}


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_output_columns_are_the_union_minus_duplicate_key(case: BackendCase) -> None:
    # flat relational record-merge: union of both sides' fields minus the duplicated key (§3.3). Both
    # payloads (left njet, right nmu) and the single key triple must survive; a normalizer reading a
    # missing column raises. A grouped/nested output cannot be read row-wise and fails here too.
    rows = _join_rows(case, how="inner")
    assert rows and all(len(r) == len(COLUMNS) for r in rows)
