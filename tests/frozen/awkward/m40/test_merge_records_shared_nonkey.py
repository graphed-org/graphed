"""M40 iter-2 — ``merge_records`` REJECTS a shared NON-key column (contract F7).

``merge_records`` zips two already-gathered blocks column-wise, dropping the ONE shared ``on`` key
(see ``test_join_primitives.test_merge_records_is_field_union_minus_duplicate_key``). But when a
NON-key column exists on BOTH sides (e.g. both tables carry ``njet``), the union is ambiguous: keep
left, keep right, or suffix. The current impl silently keeps the LEFT value and DROPS the right —
observed data loss (commit 4bc452e)::

    merge_records({__joinkey__:[1,2], njet:[10,20]}, {__joinkey__:[1,2], njet:[100,200]}, on=[__joinkey__])
    -> {__joinkey__:[1,2], njet:[10,20]}      # right njet [100,200] SILENTLY LOST, no error

The safe pinned behavior (contract F7): a shared non-key column must raise a clear ``ValueError``
rather than silently drop a side. Fails now for the right reason: no exception is raised.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from graphed.awkward import AwkwardBackend


def _block(**cols: object) -> ak.Array:
    return ak.Array(
        {k: (np.asarray(v, dtype=np.uint64) if k == "__joinkey__" else v) for k, v in cols.items()}
    )


def test_merge_records_rejects_shared_nonkey_column() -> None:
    be = AwkwardBackend()
    left = _block(__joinkey__=[1, 2], njet=[10, 20])
    right = _block(__joinkey__=[1, 2], njet=[100, 200])  # `njet` is a NON-key field present on BOTH sides
    with pytest.raises(ValueError):
        be.merge_records(left, right, on=["__joinkey__"])
