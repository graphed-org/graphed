"""M40 GAP-3 — the ``grouped`` join convenience is correctly ABSENT from the neutral/numpy path
(README pin: "The grouped shape is ONLY here (awkward post-op) ... this convenience is correctly
ABSENT on numpy"; spec INVARIANTS: "Grouped shape is ONLY `gak.join(grouped=True)`").

``grouped=True`` lives EXCLUSIVELY on ``graphed.awkward.gak.join`` (``tests/frozen/awkward/m40/
test_grouped.py``); the neutral ``graphed.join`` — the same entry point the numpy path uses — must
reject it. Right now ``graphed.join`` doesn't exist -> AttributeError (right reason); once
implemented, an impl that widens the neutral signature to silently accept (and ignore, or worse,
half-support) ``grouped`` would make this ``pytest.raises`` block report "DID NOT RAISE".
"""

from __future__ import annotations

import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.numpy import NumpyBackend, from_record


def test_grouped_is_rejected_on_the_neutral_numpy_join_path() -> None:
    s = Session(NumpyBackend())
    left = from_record(
        s,
        "l",
        run=np.array([1], dtype=np.int64),
        lumi=np.array([1], dtype=np.int64),
        event=np.array([1], dtype=np.int64),
        lv=np.array([10.0]),
    )
    right = from_record(
        s,
        "r",
        run=np.array([1], dtype=np.int64),
        lumi=np.array([1], dtype=np.int64),
        event=np.array([1], dtype=np.int64),
        rv=np.array([20.0]),
    )
    with pytest.raises(TypeError):
        graphed.join(left, right, on=["run", "lumi", "event"], how="inner", grouped=True)
    # the backend itself carries no grouped-join convenience either (that's an awkward-only post-op).
    assert not hasattr(NumpyBackend(), "join_grouped")
