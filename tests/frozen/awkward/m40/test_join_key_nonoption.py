"""M40 iteration-3 — the awkward JOIN primitives must keep the COALESCED key column NON-OPTION
(plan §3.3 null carrier: only the ABSENT side's NON-key fields become option-typed; the ``on`` key is
coalesced from whichever side is present, so it is never null and must stay non-option).

``test_join_primitives.py`` pins that a left/outer join makes the ABSENT SIDE's field an option
(``ak.is_none`` on ``rv``) but never asserts the KEY's type. On HEAD ``merge_records`` returns the
merged ``__joinkey__`` as ``?uint64`` (an option) even though it carries no nulls -- the wrong TYPE.
This is the primitive-layer witness of the frontend divergence (awkward materializes an option key,
numpy does not); numpy's primitive already keeps the key ``uint64``, so only the awkward primitive is
exercised here.

Non-vacuity: derived purely from SQL semantics (the coalesced key is never null => non-option); the
current impl produces ``?uint64`` and fails the ``"?" not in type`` assertion.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from graphed.awkward import AwkwardBackend


def _block(**cols: object) -> ak.Array:
    return ak.Array({k: (np.asarray(v, dtype=np.uint64) if k == "__joinkey__" else v) for k, v in cols.items()})


@pytest.mark.parametrize("how", ["left", "outer"])
def test_merge_records_coalesced_key_is_nonoption(how: str) -> None:
    # left key 1 has no right match (how=left/outer keeps it); the merged join key is coalesced from
    # the present side, so it is fully valid AND its TYPE must be non-option (uint64, not ?uint64).
    be = AwkwardBackend()
    left = _block(__joinkey__=[1, 2], lv=[10, 20])
    right = _block(__joinkey__=[2, 3], rv=[200, 300])
    bi, pi = be.match_indices(left, right, on=["__joinkey__"], how=how)
    merged = be.merge_records(be.take(left, bi), be.take(right, pi), on=["__joinkey__"])

    key = merged["__joinkey__"]
    assert not bool(ak.any(ak.is_none(key))), f"how={how}: no join key row may be null (keys are coalesced)"
    assert "?" not in str(key.type), (
        f"how={how}: the coalesced join key must be NON-option; got type {str(key.type)!r} "
        f"(an option key column violates §3.3 key coalescing)"
    )
