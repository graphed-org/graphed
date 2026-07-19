"""M40 iteration-4 — for an OPTION-typed ``on`` key whose null SURVIVES the join, the RECORDED op_form
and the MATERIALIZED block must AGREE on the key type, and that type must be OPTION.

This is the option-key mirror of ``test_noninner_key_nonoption.py`` (which pins that a NON-option key
stays NON-option). Plan §3.3: the coalesced key follows the INPUT key's optionality — a non-option
input can never be null (non-option out); an OPTION input carrying a genuine null on a present,
unmatched row keeps that null (option out). op_form must declare, and materialize must produce, the
same key type.

On HEAD (awkward, via the public ``graphed.join`` + ``Session.form``/``Session.materialize`` surface):
op_form declares ``k: ?int64`` (option — correct) but MATERIALIZE produces ``k: int64`` (non-option)
AND corrupts the null key value to a neighbour's (e.g. the ``[None, 2]`` left key materializes as
``[2, 2]``). Two HEAD defects, both frozen here: (1) op_form vs materialized DISAGREE on the key type;
(2) the surviving null is lost — no row with a null key, though the SQL oracle keeps one.

Expected values follow only from SQL/pandas COALESCE semantics (an option input's surviving null =>
option key, null preserved), never from the awkward join kernel. numpy has no ``.tt`` typetracer form
surface, so the op_form-vs-materialized agreement is asserted on the awkward backend only (numpy value
parity is covered at the primitive layer in ``tests/frozen/awkward/m40/test_join_null_key.py``).
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward


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


# how values where the null key is on the PRESENT-and-kept side, so it SURVIVES into the result and the
# coalesced key must be option. (how=right with a null on the LEFT drops the null row, so it is not a
# surviving-null case and is covered for values in the primitive suite instead.)
@pytest.mark.parametrize("how", ["left", "outer"])
def test_awkward_option_null_key_form_matches_materialized_and_stays_option(how: str) -> None:
    s = Session(AwkwardBackend())
    # left key [None, 2]: the None row is present but unmatched (right has only key 2) -> it survives
    # left/outer with a null key. SQL oracle rows: {(None, 20, None), (2, 21, 200)}.
    left = _option_key_source(s, "left", [None, 2], lv=[20, 21])
    right = _option_key_source(s, "right", [2], rv=[200])
    j = graphed.join(left, right, on=["k"], how=how)

    declared = _leaf(s.form(j).tt["k"])  # recorded op_form key type
    block = s.materialize(j)
    produced = _leaf(block["k"])

    assert "?" in produced, (
        f"how={how}: materialized key is {produced!r} (non-option); a surviving null key must be "
        f"OPTION-typed — the coalesced key drops the null the SQL oracle keeps"
    )
    assert declared == produced, (
        f"how={how}: op_form and materialized block DISAGREE on key type: "
        f"op_form={declared!r} vs materialized={produced!r}"
    )
    keys = [r.get("k") for r in ak.to_list(block)]
    assert None in keys, (
        f"how={how}: the surviving null-key row was lost/corrupted — materialized keys {keys} carry no "
        f"null, but the SQL oracle keeps the unmatched None-key left row"
    )
