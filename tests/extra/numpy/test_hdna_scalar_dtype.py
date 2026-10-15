"""``np.where``/``np.clip`` scalars on a deferred numpy array keep the dtype numpy gives them (GW4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from graphed import Session
from graphed.numpy import NumpyBackend, from_array

I32: npt.NDArray[np.int32] = np.arange(6, dtype=np.int32)
F32: npt.NDArray[np.float32] = np.arange(6, dtype=np.float32)


@pytest.mark.parametrize("scalar", [0, 1.5, np.int32(1), np.float32(1), np.int64(1)], ids=repr)
@pytest.mark.parametrize("arr", [I32, F32], ids=["i32", "f32"])
@pytest.mark.parametrize(
    "op",
    [
        lambda x, s: np.where(x > 2, x, s),
        lambda x, s: np.where(x > 2, s, x),
        lambda x, s: np.clip(x, s, 4),
        lambda x, s: np.clip(x, None, s),
    ],
    ids=["where_right", "where_left", "clip_low", "clip_high"],
)
def test_a_scalar_operand_keeps_its_numpy_dtype(op: Callable[[Any, Any], Any], arr: Any, scalar: Any) -> None:
    session = Session(NumpyBackend())
    got = np.asarray(session.materialize(op(from_array(session, "x", arr), scalar)))
    want = op(arr, scalar)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)
