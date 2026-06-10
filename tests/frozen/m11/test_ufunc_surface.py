"""M11: the full single-output numpy-ufunc tier records canonical, backend-agnostic ops.

dask.array parity P0.2 (see dask-array-parity-plan.md): every single-output numpy ufunc a user can
apply to a deferred array must record one canonical op via ``__array_ufunc__`` — graphed itself
never imports numpy (plan §A.4); dispatch is by ufunc name only.
"""

from __future__ import annotations

import numpy as np
import pytest
from m11_toy import ToyBackend, source

from graphed import Session

UNARY = [
    (np.exp, "exp"),
    (np.exp2, "exp2"),
    (np.expm1, "expm1"),
    (np.log, "log"),
    (np.log1p, "log1p"),
    (np.log2, "log2"),
    (np.log10, "log10"),
    (np.cbrt, "cbrt"),
    (np.square, "square"),
    (np.reciprocal, "reciprocal"),
    (np.sign, "sign"),
    (np.signbit, "signbit"),
    (np.floor, "floor"),
    (np.ceil, "ceil"),
    (np.trunc, "trunc"),
    (np.rint, "rint"),
    (np.fabs, "fabs"),
    (np.conjugate, "conj"),
    (np.isnan, "isnan"),
    (np.isinf, "isinf"),
    (np.isfinite, "isfinite"),
    (np.logical_not, "logical_not"),
    (np.tan, "tan"),
    (np.tanh, "tanh"),
    (np.arcsin, "arcsin"),
    (np.arccos, "arccos"),
    (np.arctan, "arctan"),
    (np.arcsinh, "arcsinh"),
    (np.arccosh, "arccosh"),
    (np.arctanh, "arctanh"),
    (np.deg2rad, "deg2rad"),
    (np.rad2deg, "rad2deg"),
    (np.spacing, "spacing"),
    (np.positive, "pos"),
    (np.invert, "invert"),
]

BINARY = [
    (np.arctan2, "arctan2"),
    (np.copysign, "copysign"),
    (np.nextafter, "nextafter"),
    (np.ldexp, "ldexp"),
    (np.fmod, "fmod"),
    (np.fmax, "fmax"),
    (np.fmin, "fmin"),
    (np.floor_divide, "floordiv"),
    (np.remainder, "mod"),
    (np.logaddexp, "logaddexp"),
    (np.logaddexp2, "logaddexp2"),
    (np.float_power, "float_power"),
    (np.heaviside, "heaviside"),
    (np.gcd, "gcd"),
    (np.lcm, "lcm"),
    (np.bitwise_and, "and"),
    (np.bitwise_or, "or"),
    (np.bitwise_xor, "xor"),
    (np.left_shift, "lshift"),
    (np.right_shift, "rshift"),
    (np.logical_and, "logical_and"),
    (np.logical_or, "logical_or"),
    (np.logical_xor, "logical_xor"),
]


@pytest.mark.parametrize(("fn", "op"), UNARY, ids=[op for _, op in UNARY])
def test_unary_ufunc_records_canonical_op(fn: object, op: str) -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert s.form(fn(a)).describe() == op  # type: ignore[operator]


@pytest.mark.parametrize(("fn", "op"), BINARY, ids=[op for _, op in BINARY])
def test_binary_ufunc_records_canonical_op(fn: object, op: str) -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    b = source(s, "b")
    assert s.form(fn(a, b)).describe() == op  # type: ignore[operator]


def test_binary_ufunc_with_scalar_and_reflected() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert s.form(np.fmax(a, 1.0)).describe() == "fmax"
    assert s.form(np.fmax(1.0, a)).describe() == "fmax"  # reflected: scalar first
    assert s.form(np.copysign(a, -1.0)).describe() == "copysign"


def test_alias_ufuncs_intern_to_the_same_node() -> None:
    # numpy aliases must record the SAME canonical op, so hash-consing dedups them
    s = Session(ToyBackend())
    a = source(s, "a")
    assert np.degrees(a).node_id == np.rad2deg(a).node_id
    assert np.radians(a).node_id == np.deg2rad(a).node_id
    assert np.divide(a, a).node_id == (a / a).node_id
    assert np.absolute(a).node_id == abs(a).node_id


def test_repeated_ufunc_interns_to_zero_new_nodes() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    np.exp(a)
    n = s.node_count()
    np.exp(a)
    assert s.node_count() == n


def test_new_operator_dunders_record_canonical_ops() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    b = source(s, "b")
    assert s.form(a // b).describe() == "floordiv"
    assert s.form(7 // a).describe() == "floordiv"  # reflected
    assert s.form(a ^ b).describe() == "xor"
    assert s.form(a << b).describe() == "lshift"
    assert s.form(a >> b).describe() == "rshift"
    assert s.form(a << 2).describe() == "lshift"
    assert s.form(+a).describe() == "pos"
    assert s.form(2**a).describe() == "power"  # reflected pow
    assert s.form(3 & a).describe() == "and"  # reflected boolean
    assert s.form(3 | a).describe() == "or"
    assert s.form(3 ^ a).describe() == "xor"
