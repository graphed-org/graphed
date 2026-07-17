"""Cover the NumpyBackend op surface and its type-checking / boundary methods (M2)."""

from __future__ import annotations

import numpy as np
import pytest

from graphed import GraphedTypeError, Session
from graphed.numpy import NumpyBackend, from_array


def test_sub_and_div_arith() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([10.0, 20.0]))
    b = from_array(s, "b", np.array([2.0, 5.0]))
    assert np.allclose(s.materialize(a - b), [8.0, 15.0])
    assert np.allclose(s.materialize(a / b), [5.0, 4.0])


def test_arith_requires_numeric() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    text = from_array(s, "t", np.array(["x", "y"]))
    with pytest.raises(GraphedTypeError):
        a + text


def test_filter_requires_bool_mask() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    bad = from_array(s, "bad", np.array([1, 0]))  # int mask
    with pytest.raises(GraphedTypeError):
        a.filter(bad)


def test_sum_requires_numeric() -> None:
    s = Session(NumpyBackend())
    text = from_array(s, "t", np.array(["x", "y"]))
    with pytest.raises(GraphedTypeError):
        text.reduce("sum")


def test_unsupported_op_form_and_eval() -> None:
    backend = NumpyBackend()
    with pytest.raises(TypeError):
        backend.op_form("nope", [], {})
    with pytest.raises(TypeError):
        backend.eval_stage("nope", [], {})


def test_boundary_ops_and_project() -> None:
    backend = NumpyBackend()
    assert "sum" in backend.boundary_ops()
    assert "map" in backend.boundary_ops()
    assert backend.project("filter", "USED", {}) == "USED"
