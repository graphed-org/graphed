"""M40 iteration-5 — the RIGHT-side mirror of ``test_noninner_null_key_option.py``: for an OPTION-typed
``on`` key whose null SURVIVES on the RIGHT side, the RECORDED op_form must AGREE with the MATERIALIZED
block on the key type, and both must be OPTION.

``test_noninner_null_key_option.py`` covers only a LEFT option key (how ∈ {left, outer}) and explicitly
excludes this right/non-option mirror. It pins the case where op_form happens to be correct (option) but
MATERIALIZE was wrong. THIS file pins the opposite, uncovered leg: op_form is WRONG (under-declares) while
materialize is correct.

Setup (the surviving null lives on the RIGHT): LEFT key column NON-option ``int64``; RIGHT key column
OPTION ``?int64`` carrying a genuine null on a right-only, unmatched row. In how ∈ {right, outer} that
right-only null row is kept (SQL right/outer keeps every right row), so its coalesced ``on`` key is null.

Contract (§A.3.1: the serialized IR / recorded op_form is the CANONICAL DURABLE representation — it must
not lie about a column's type; §3.3: a right/outer join keeps the unmatched right row, and the coalesced
key follows the INPUT key's optionality). General principle encoded here: for a coalesced ``on`` key,
op_form is OPTION iff EITHER input key is option-typed — the surviving side's null can appear. Non-option
on BOTH sides stays non-option. So op_form must NOT UNDER-declare nullability.

On HEAD (7cb8d41, measured through the public ``graphed.join`` + ``Session.form``/``Session.materialize``
surface, awkward backend): ``Session.form(j).tt["k"]`` declares ``k: int64`` (WRONG — under-declares) but
``Session.materialize(j)["k"]`` produces ``k: ?int64`` carrying a real ``None`` (CORRECT). measured keys
= ``[5, None]``. The recorded durable form says the key can't be null while the durable data HAS one — an
unsound recorded form (a null-key trap in the durable record itself, §A.3.1).

Expected values follow only from SQL/pandas COALESCE semantics (an option input's surviving null =>
option key, null preserved) and the pandas value oracle — never from the awkward join kernel / op_form.
numpy Form has no ``.tt`` typetracer surface, so the op_form-vs-materialized agreement is asserted on the
awkward backend only.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward

_USER_COLS = ("k", "lv", "rv")


def _leaf(arr: object) -> str:
    """Per-column leaf type without the length prefix: ``2 * ?int64`` -> ``?int64``."""
    return str(arr.type).split("*")[-1].strip()  # type: ignore[attr-defined]


def _option_key_source(session: Session, name: str, k: list[object], **payload: list[int]) -> object:
    """A recorded source whose ``k`` column is OPTION-typed (``?int64``) with a genuine null."""
    data = np.array([10_000_000 if v is None else v for v in k], dtype=np.int64)
    mask = np.array([v is None for v in k], dtype=bool)
    cols: dict[str, object] = {"k": ak.Array(np.ma.MaskedArray(data, mask=mask))}
    cols.update({f: np.asarray(v, dtype=np.int64) for f, v in payload.items()})
    return from_awkward(session, name, ak.Array(cols))


def _nonoption_key_source(session: Session, name: str, k: list[int], **payload: list[int]) -> object:
    """A recorded source whose ``k`` column is a plain NON-option ``int64`` (no null carrier)."""
    cols: dict[str, object] = {"k": np.asarray(k, dtype=np.int64)}
    cols.update({f: np.asarray(v, dtype=np.int64) for f, v in payload.items()})
    return from_awkward(session, name, ak.Array(cols))


def _srt(rows: list[tuple[int | None, ...]]) -> list[tuple[int | None, ...]]:
    # None-safe multiset sort (can't order None against int directly).
    return sorted(rows, key=lambda r: tuple((x is None, -1 if x is None else x) for x in r))


def _block_rows(block: object) -> list[tuple[int | None, ...]]:
    """Materialized awkward join block -> sorted multiset of ``_USER_COLS`` rows, nulls kept as ``None``
    (the internal ``__joinkey__`` column is ignored — it is not a user column)."""
    return _srt([tuple(r.get(c) for c in _USER_COLS) for r in ak.to_list(block)])  # type: ignore[union-attr]


def _oracle(
    left: dict[str, list[object]], right: dict[str, list[object]], *, how: str
) -> list[tuple[int | None, ...]]:
    """The NULL-preserving, KEY-COALESCING reference: ``pandas.merge(how=...)`` over nullable ``Int64``
    (so a null key stays a null integer, and a missing non-key cell reads ``<NA>`` -> ``None``)."""
    ldf = pd.DataFrame({c: pd.array(v, dtype="Int64") for c, v in left.items()})
    rdf = pd.DataFrame({c: pd.array(v, dtype="Int64") for c, v in right.items()})
    ref = pd.merge(ldf, rdf, on=["k"], how=how)
    return _srt(
        [
            tuple(None if pd.isna(ref[c].iloc[i]) else int(ref[c].iloc[i]) for c in _USER_COLS)
            for i in range(len(ref))
        ]
    )


# The null key is on the RIGHT (present-and-kept in right/outer), so it SURVIVES into the result and the
# coalesced key must be option. (how=left would drop the right-only null row entirely — not a surviving
# case — so left is excluded here, mirroring the LEFT-key file's {left, outer}.)
@pytest.mark.parametrize("how", ["right", "outer"])
def test_awkward_right_option_null_key_form_matches_materialized_and_stays_option(how: str) -> None:
    s = Session(AwkwardBackend())
    # left key [5] non-option; right key [5, None] option, the None on a right-only unmatched row that
    # right/outer keeps -> the null survives. SQL oracle rows: {(5, 50, 500), (None, None, 999)}.
    left_cols: dict[str, list[object]] = {"k": [5], "lv": [50]}
    right_cols: dict[str, list[object]] = {"k": [5, None], "rv": [500, 999]}
    left = _nonoption_key_source(s, "left", [5], lv=[50])
    right = _option_key_source(s, "right", [5, None], rv=[500, 999])
    j = graphed.join(left, right, on=["k"], how=how)

    declared = _leaf(s.form(j).tt["k"])  # recorded op_form key type (the CANONICAL durable form, §A.3.1)
    block = s.materialize(j)
    produced = _leaf(block["k"])

    assert "?" in produced, (
        f"how={how}: materialized key is {produced!r} (non-option); the surviving right-only null key "
        f"must be OPTION-typed"
    )
    assert "?" in declared, (
        f"how={how}: recorded op_form UNDER-declares the key as {declared!r} (non-option) — an option "
        f"input key whose null survives makes the coalesced key OPTION; the durable form must not claim "
        f"the key can never be null while materialize produces one"
    )
    assert declared == produced, (
        f"how={how}: recorded op_form and materialized block DISAGREE on key type: "
        f"op_form={declared!r} vs materialized={produced!r}"
    )

    keys = [r.get("k") for r in ak.to_list(block)]
    assert None in keys, (
        f"how={how}: the surviving null-key row is absent — materialized keys {keys} carry no null, but "
        f"the SQL oracle keeps the unmatched right None-key row"
    )
    got = _block_rows(block)
    expected = _oracle(left_cols, right_cols, how=how)
    assert got == expected, (
        f"how={how}: materialized rows must equal the null-preserving key-coalescing pandas oracle\n"
        f"  got={got}\n  exp={expected}"
    )


# Completeness control (guards against an over-correction that makes EVERY coalesced key option): with a
# NON-option key on BOTH sides — right-only orphan key 7 is a real value, not a null — op_form AND
# materialize must both stay NON-option. Passes on HEAD; must stay green after the fix.
@pytest.mark.parametrize("how", ["right", "outer"])
def test_awkward_both_nonoption_key_stays_nonoption(how: str) -> None:
    s = Session(AwkwardBackend())
    left = _nonoption_key_source(s, "left", [5], lv=[50])
    right = _nonoption_key_source(s, "right", [5, 7], rv=[500, 700])
    j = graphed.join(left, right, on=["k"], how=how)

    declared = _leaf(s.form(j).tt["k"])
    block = s.materialize(j)
    produced = _leaf(block["k"])

    assert "?" not in produced, f"how={how}: both-non-option key materialized as option ({produced!r})"
    assert "?" not in declared, f"how={how}: both-non-option key op_form is option ({declared!r})"
    assert declared == produced, (
        f"how={how}: op_form and materialized disagree on a both-non-option key: "
        f"op_form={declared!r} vs materialized={produced!r}"
    )
