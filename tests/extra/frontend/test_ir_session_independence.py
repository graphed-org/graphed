"""Optimized IR is a function of the outputs' cone, not of the rest of the Session (#105)."""

from __future__ import annotations

import numpy as np
import pytest

from graphed import Array, Session, compile_ir
from graphed.numpy import NumpyBackend, from_array


def _masked(mask_first: bool, unrelated: bool, incremental: bool) -> tuple[Session, Array]:
    s = Session(NumpyBackend(), incremental=incremental)
    x = from_array(s, "x", np.arange(6.0))
    if unrelated:
        _ = from_array(s, "b", np.arange(3.0)) * 2
        _ = x - 3.0
    if mask_first:
        _ = x > 1.0
    return s, (x * 2)[x > 1.0]


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize(("mask_first", "unrelated"), [(True, False), (False, True), (True, True)])
def test_optimized_ir_matches_a_fresh_session(
    mask_first: bool, unrelated: bool, incremental: bool
) -> None:
    fresh_s, fresh = _masked(False, False, incremental)
    s, out = _masked(mask_first, unrelated, incremental)
    assert s.serialized_ir(out) == fresh_s.serialized_ir(fresh)
    assert compile_ir(s, out).ir == compile_ir(fresh_s, fresh).ir


def test_a_different_cone_serializes_differently() -> None:
    s, out = _masked(False, False, False)
    assert s.serialized_ir(out) != s.serialized_ir(out * 3)
