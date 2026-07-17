"""M11: the full elementwise tier evaluates exactly as numpy does (dask.array parity P0.2).

Every canonical op the frontend records must be implemented by ``NumpyBackend`` — graphed-numpy
previously covered only 4 of the ~25 recordable ops. Reference semantics ARE numpy's: each case
materializes the deferred program and compares against applying the same ufunc directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import Session

from graphed_numpy import NumpyBackend, from_array

F = np.array([0.25, 0.5, 1.5, 2.0])  # positive floats: safe for log/sqrt/reciprocal
G = np.array([2.0, 0.5, 3.0, 1.5])
SYM = np.array([-0.9, -0.1, 0.4, 0.8])  # |x| < 1: safe for arcsin/arccos/arctanh
GE1 = np.array([1.0, 1.5, 2.5, 9.0])  # >= 1: safe for arccosh
IA = np.array([12, 5, 7, 9], dtype=np.int64)
JA = np.array([3, 2, 1, 4], dtype=np.int64)
B = np.array([True, False, True, False])
C = np.array([True, True, False, False])

UNARY_CASES = [
    (np.exp, F),
    (np.exp2, F),
    (np.expm1, F),
    (np.log, F),
    (np.log1p, F),
    (np.log2, F),
    (np.log10, F),
    (np.sqrt, F),
    (np.cbrt, F),
    (np.square, F),
    (np.reciprocal, F),
    (np.sign, SYM),
    (np.signbit, SYM),
    (np.floor, G),
    (np.ceil, G),
    (np.trunc, G),
    (np.rint, G),
    (np.fabs, SYM),
    (np.conjugate, F),
    (np.isnan, F),
    (np.isinf, F),
    (np.isfinite, F),
    (np.logical_not, B),
    (np.sin, F),
    (np.cos, F),
    (np.tan, F),
    (np.sinh, F),
    (np.cosh, F),
    (np.tanh, F),
    (np.arcsin, SYM),
    (np.arccos, SYM),
    (np.arctan, F),
    (np.arcsinh, F),
    (np.arccosh, GE1),
    (np.arctanh, SYM),
    (np.deg2rad, F),
    (np.rad2deg, F),
    (np.spacing, F),
    (np.positive, F),
    (np.negative, F),
    (np.absolute, SYM),
    (np.invert, IA),
]

BINARY_CASES = [
    (np.add, F, G),
    (np.subtract, F, G),
    (np.multiply, F, G),
    (np.true_divide, F, G),
    (np.arctan2, F, G),
    (np.hypot, F, G),
    (np.copysign, F, SYM),
    (np.nextafter, F, G),
    (np.ldexp, F, JA),
    (np.fmod, F, G),
    (np.fmax, F, G),
    (np.fmin, F, G),
    (np.floor_divide, F, G),
    (np.remainder, F, G),
    (np.power, F, G),
    (np.float_power, F, G),
    (np.logaddexp, F, G),
    (np.logaddexp2, F, G),
    (np.heaviside, SYM, G),
    (np.maximum, F, G),
    (np.minimum, F, G),
    (np.gcd, IA, JA),
    (np.lcm, IA, JA),
    (np.bitwise_and, IA, JA),
    (np.bitwise_or, IA, JA),
    (np.bitwise_xor, IA, JA),
    (np.left_shift, IA, JA),
    (np.right_shift, IA, JA),
    (np.logical_and, B, C),
    (np.logical_or, B, C),
    (np.logical_xor, B, C),
]


@pytest.mark.parametrize(("fn", "data"), UNARY_CASES, ids=[f.__name__ for f, _ in UNARY_CASES])
def test_unary_matches_numpy(fn: np.ufunc, data: np.ndarray) -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", data)
    np.testing.assert_array_equal(np.asarray(s.materialize(fn(a))), fn(data))


@pytest.mark.parametrize(("fn", "x", "y"), BINARY_CASES, ids=[f.__name__ for f, *_ in BINARY_CASES])
def test_binary_matches_numpy(fn: np.ufunc, x: np.ndarray, y: np.ndarray) -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", x)
    b = from_array(s, "b", y)
    np.testing.assert_array_equal(np.asarray(s.materialize(fn(a, b))), fn(x, y))


@pytest.mark.parametrize(("fn", "x", "y"), BINARY_CASES, ids=[f.__name__ for f, *_ in BINARY_CASES])
def test_binary_with_scalar_matches_numpy(fn: np.ufunc, x: np.ndarray, y: np.ndarray) -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", x)
    scalar = y[0].item()
    np.testing.assert_array_equal(np.asarray(s.materialize(fn(a, scalar))), fn(x, scalar))
    if fn is not np.ldexp:  # numpy itself rejects the reflected order ldexp(int, float-array)
        np.testing.assert_array_equal(np.asarray(s.materialize(fn(scalar, a))), fn(scalar, x))


def test_comparisons_materialize_to_bool_arrays() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", F)
    b = from_array(s, "b", G)
    np.testing.assert_array_equal(np.asarray(s.materialize(a > b)), F > G)
    np.testing.assert_array_equal(np.asarray(s.materialize(a <= b)), F <= G)
    np.testing.assert_array_equal(np.asarray(s.materialize(a == b)), F == G)
    np.testing.assert_array_equal(np.asarray(s.materialize(a != b)), F != G)
    np.testing.assert_array_equal(np.asarray(s.materialize((a > 0.4) & (b < 2.5))), (F > 0.4) & (G < 2.5))


def test_operator_dunders_match_numpy() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", F)
    i = from_array(s, "i", IA)
    np.testing.assert_array_equal(np.asarray(s.materialize(a // 0.5)), F // 0.5)
    np.testing.assert_array_equal(np.asarray(s.materialize(7 // a)), 7 // F)
    np.testing.assert_array_equal(np.asarray(s.materialize(i ^ 3)), IA ^ 3)
    np.testing.assert_array_equal(np.asarray(s.materialize(i << 2)), IA << 2)
    np.testing.assert_array_equal(np.asarray(s.materialize(i >> 1)), IA >> 1)
    np.testing.assert_array_equal(np.asarray(s.materialize(+a)), +F)
    np.testing.assert_array_equal(np.asarray(s.materialize(2.0**a)), 2.0**F)


def test_np_sum_dispatches_and_matches_numpy() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", F)
    assert float(np.asarray(s.materialize(np.sum(a)))) == pytest.approx(float(np.sum(F)))
