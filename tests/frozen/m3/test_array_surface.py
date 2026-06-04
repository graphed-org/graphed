"""The M3 awkward op surface on Array: field access, indexing, comparisons, ufuncs, scalars."""

from __future__ import annotations

import numpy as np
import pytest
from toy import ToyBackend, source

from graphed import Session


def test_field_access_records_field_op() -> None:
    s = Session(ToyBackend())
    events = source(s, "events")
    pt = events.Muon.pt  # two field ops
    assert s.form(pt).describe() == "field"
    assert s.node_count() == 3  # events, Muon, pt


def test_getitem_mask_field_and_error() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    mask = source(s, "m")
    assert s.form(a[mask]).describe() == "getitem"
    assert s.form(a["field"]).describe() == "field"
    with pytest.raises(TypeError):
        _ = a[1.5]


def test_arithmetic_with_scalar_and_array() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    b = source(s, "b")
    assert s.form(a + b).describe() == "add"
    assert s.form(a * 2).describe() == "mul"
    assert s.form(2 * a).describe() == "mul"  # reflected
    assert s.form(1 - a).describe() == "sub"  # reflected
    assert s.form(a / b).describe() == "div"
    assert s.form(a % 3).describe() == "mod"
    assert s.form(a**2).describe() == "power"
    assert s.form(-a).describe() == "neg"
    assert s.form(abs(a)).describe() == "abs"


def test_comparisons_and_boolean_ops() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    b = source(s, "b")
    assert s.form(a > 5).describe() == "gt"
    assert s.form(a < b).describe() == "lt"
    assert s.form(a >= 0).describe() == "ge"
    assert s.form(a <= b).describe() == "le"
    assert s.form(a == b).describe() == "eq"
    assert s.form(a != b).describe() == "ne"
    assert s.form((a > 1) & (b < 2)).describe() == "and"
    assert s.form((a > 1) | (b < 2)).describe() == "or"
    assert s.form(~(a > 1)).describe() == "invert"


def test_numpy_ufuncs_record_canonical_ops() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    b = source(s, "b")
    assert s.form(np.sqrt(a)).describe() == "sqrt"
    assert s.form(np.cos(a)).describe() == "cos"
    assert s.form(np.cosh(a)).describe() == "cosh"
    assert s.form(np.hypot(a, b)).describe() == "hypot"
    assert s.form(np.maximum(a, b)).describe() == "maximum"
    assert s.form(np.subtract(3, a)).describe() == "sub"  # scalar-left via ufunc


def test_unhashable_due_to_deferred_eq() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        hash(a)


def test_unsupported_scalar_operand_rejected() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        _ = a + object()
