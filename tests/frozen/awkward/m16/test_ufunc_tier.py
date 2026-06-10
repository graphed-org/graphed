"""M16.1: the full M11 canonical ufunc tier evaluates on jagged arrays (parity plan P0).

The frontend records ~85 canonical ops; the awkward backend previously implemented 25. Every
case applies the numpy ufunc to a DEFERRED jagged array (recording one canonical op with a
typetracer form) and compares the materialization against the same ufunc on the raw array."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from m16_helpers import session_events

from graphed_awkward import from_awkward

F = [[0.25, 0.5], [], [1.5, 2.0, 0.75], [1.0]]  # positive: safe for log/sqrt
SYM = [[-0.9, -0.1], [], [0.4, 0.8, -0.5], [0.3]]  # |x| < 1: safe for arcsin/arctanh
GE1 = [[1.0, 1.5], [], [2.5, 9.0, 4.0], [1.1]]  # >= 1: safe for arccosh
INTS = [[12, 5], [], [7, 9, 3], [4]]
BOOLS = [[True, False], [], [True, True, False], [False]]

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
    (np.floor, F),
    (np.ceil, F),
    (np.trunc, F),
    (np.rint, F),
    (np.fabs, SYM),
    (np.conjugate, F),
    (np.isnan, F),
    (np.isinf, F),
    (np.isfinite, F),
    (np.logical_not, BOOLS),
    (np.tan, F),
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
    (np.invert, INTS),
]

BINARY_CASES = [
    (np.arctan2, F, GE1),
    (np.copysign, F, SYM),
    (np.nextafter, F, GE1),
    (np.fmod, F, GE1),
    (np.fmax, F, GE1),
    (np.fmin, F, GE1),
    (np.floor_divide, F, GE1),
    (np.remainder, F, GE1),
    (np.logaddexp, F, GE1),
    (np.logaddexp2, F, GE1),
    (np.float_power, F, GE1),
    (np.heaviside, SYM, GE1),
    (np.gcd, INTS, INTS),
    (np.lcm, INTS, INTS),
    (np.bitwise_xor, INTS, INTS),
    (np.left_shift, INTS, INTS),
    (np.right_shift, INTS, INTS),
    (np.logical_and, BOOLS, BOOLS),
    (np.logical_or, BOOLS, BOOLS),
    (np.logical_xor, BOOLS, BOOLS),
]


def _src(s, name, data):  # type: ignore[no-untyped-def]
    return from_awkward(s, name, ak.Array(data))


@pytest.mark.parametrize(("fn", "data"), UNARY_CASES, ids=[f.__name__ for f, _ in UNARY_CASES])
def test_unary_ufunc_on_jagged_matches_awkward(fn, data) -> None:  # type: ignore[no-untyped-def]
    s, _, _ = session_events()
    a = _src(s, "a", data)
    deferred = fn(a)
    assert s.form(deferred).is_typetracer  # record-time inference succeeded, metadata only
    got = ak.Array(s.materialize(deferred))
    ref = fn(ak.Array(data))
    assert ak.array_equal(got, ref, equal_nan=True)  # same kernel, same inputs: exactly equal


@pytest.mark.parametrize(("fn", "x", "y"), BINARY_CASES, ids=[f.__name__ for f, *_ in BINARY_CASES])
def test_binary_ufunc_on_jagged_matches_awkward(fn, x, y) -> None:  # type: ignore[no-untyped-def]
    s, _, _ = session_events()
    a = _src(s, "a", x)
    b = _src(s, "b", y)
    got = ak.Array(s.materialize(fn(a, b)))
    ref = fn(ak.Array(x), ak.Array(y))
    assert ak.array_equal(got, ref, equal_nan=True)


def test_scalar_operands_still_work_through_the_new_tier() -> None:
    s, g, raw = session_events()
    got = ak.Array(s.materialize(np.fmax(g.Jet.pt, 35.0)))
    assert ak.to_list(got) == ak.to_list(np.fmax(raw.Jet.pt, 35.0))
    got = ak.Array(s.materialize(np.copysign(-1.0, g.Jet.eta)))
    assert ak.to_list(got) == ak.to_list(np.copysign(-1.0, raw.Jet.eta))
