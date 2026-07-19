"""M40 — the awkward ``ShuffleBackend`` JOIN half + ``op_form("join")`` (plan §3.0, §3.3, §2.5).

``AwkwardBackend`` gains the relational join primitives the generic radix-hash kernel calls:
``match_indices(build, probe, *, on, how) -> (Index, Index)``, ``take(block, index) -> Block``,
``merge_records(left, right, *, on) -> Block``. Semantics are RELATIONAL (SQL/pandas ``merge``):
a probe row with k matches yields k output rows — NOT a list-of-matches, NOT a grouped shape
(that is only ``gak.join(grouped=True)``, tested in ``test_grouped.py``).

Load-bearing witnesses here:
  * relational DUPLICATION (k matches -> k rows) — kills a list-of-matches / first-match baseline;
  * ``merge_records`` = flat field-union MINUS the duplicate key — kills a keep-both-keys impl;
  * (a3) ``how="left"`` missing side READS a real awkward OPTION (``ak.is_none`` True), asserted as
    VALIDITY not row count — kills the ``take(-1)`` trap and its awkward analogue (naive fancy
    indexing, where index ``-1`` silently reads the LAST row instead of a null);
  * ``op_form("join")`` = flat relational record-merge form (union minus dup key; left/outer ⇒
    option-typed) — kills an identity/passthrough ``op_form`` that returns ``inputs[0]``.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward


def _block(**cols: object) -> ak.Array:
    """A flat record block; ``__joinkey__`` is the packed-u64 key column the engine routes on."""
    return ak.Array(
        {k: (np.asarray(v, dtype=np.uint64) if k == "__joinkey__" else v) for k, v in cols.items()}
    )


def test_inner_join_primitives_duplicate_relationally() -> None:
    # (a) THE relational pin: key 2 has 2 build rows x 2 probe rows -> 4 output rows (SQL merge),
    # NOT 2 (list-of-matches / grouped / first-match). Order-independent multiset assertion.
    be = AwkwardBackend()
    left = _block(__joinkey__=[1, 2, 2], lv=[10, 20, 21])
    right = _block(__joinkey__=[2, 2], rv=[200, 201])
    bi, pi = be.match_indices(left, right, on=["__joinkey__"], how="inner")
    merged = be.merge_records(be.take(left, bi), be.take(right, pi), on=["__joinkey__"])
    assert len(merged) == 4, "k=2 x k=2 must duplicate to 4 rows (relational), not collapse to 2"
    assert all(k == 2 for k in merged["__joinkey__"].to_list()), "only the matched key survives inner"
    assert set(merged.fields) == {"__joinkey__", "lv", "rv"}, "union of fields minus the duplicate key"
    pairs = sorted(zip(merged["lv"].to_list(), merged["rv"].to_list(), strict=True))
    assert pairs == sorted([(20, 200), (20, 201), (21, 200), (21, 201)])


def test_merge_records_is_field_union_minus_duplicate_key() -> None:
    # merge_records zips two already-gathered, equal-length blocks column-wise and drops the ONE
    # shared key column (not two keys, not a suffixed key). Discriminates a keep-both-keys impl.
    be = AwkwardBackend()
    left = _block(__joinkey__=[1, 2], lv=[10, 20])
    right = _block(__joinkey__=[1, 2], rv=[100, 200])
    m = be.merge_records(left, right, on=["__joinkey__"])
    assert set(m.fields) == {"__joinkey__", "lv", "rv"}
    assert m["__joinkey__"].to_list() == [1, 2]
    assert m["lv"].to_list() == [10, 20]
    assert m["rv"].to_list() == [100, 200]


def test_left_join_missing_side_reads_as_an_awkward_option_null() -> None:
    # (a3) TRAP 1 (awkward analogue of np.take(-1)): the left-only row's right field must READ NULL
    # (ak.is_none True), asserted as VALIDITY. A naive `take` (np.take, or `block[index]` where
    # index=-1 does negative indexing) gathers the LAST right row (rv=200) -> is_none False -> FAILS.
    be = AwkwardBackend()
    left = _block(__joinkey__=[1, 2], lv=[10, 20])
    right = _block(__joinkey__=[2], rv=[200])  # key 1 has NO right match
    bi, pi = be.match_indices(left, right, on=["__joinkey__"], how="left")
    merged = be.merge_records(be.take(left, bi), be.take(right, pi), on=["__joinkey__"])
    assert len(merged) == 2, "left join keeps every left row (inner would drop key 1)"
    jk = merged["__joinkey__"].to_list()
    rv_valid = ak.is_none(merged["rv"]).to_list()
    i_miss, i_hit = jk.index(1), jk.index(2)
    assert rv_valid[i_miss], "left-only row: the right field is genuinely MISSING (option null)"
    assert not rv_valid[i_hit], "the matched row is present"
    assert merged["rv"][i_hit] == 200


def test_op_form_join_is_a_flat_record_merge_union_minus_key() -> None:
    # op_form("join") builds the OUTPUT form by relational record-merge: union of both sides' fields
    # minus the duplicate key. An identity/passthrough op_form (return inputs[0]) yields only
    # {__joinkey__, lv} and FAILS the union assertion.
    s = Session(AwkwardBackend())
    be = s.backend
    left = from_awkward(s, "l", _block(__joinkey__=[1, 2], lv=[10, 20]))
    right = from_awkward(s, "r", _block(__joinkey__=[1, 2], rv=[100, 200]))
    lf, rf = s.form(left), s.form(right)

    inner = be.op_form("join", [lf, rf], {"how": "inner", "on": "__joinkey__"})
    assert set(inner.tt.fields) == {"__joinkey__", "lv", "rv"}
    assert "?" not in inner.describe(), "inner introduces no nulls -> no option in the form"


def test_op_form_left_join_makes_the_outer_side_option_typed() -> None:
    # (a3, form level) how="left"/"outer" ⇒ the right-side field is option-typed in the form (a "?"
    # in the type string). An impl that ignores `how` and builds a non-option merge FAILS.
    s = Session(AwkwardBackend())
    be = s.backend
    left = from_awkward(s, "l", _block(__joinkey__=[1, 2], lv=[10, 20]))
    right = from_awkward(s, "r", _block(__joinkey__=[1, 2], rv=[100, 200]))
    lf, rf = s.form(left), s.form(right)

    left_form = be.op_form("join", [lf, rf], {"how": "left", "on": "__joinkey__"})
    assert set(left_form.tt.fields) == {"__joinkey__", "lv", "rv"}
    assert "?" in left_form.describe(), "left join: the missing-side field becomes an awkward option"
