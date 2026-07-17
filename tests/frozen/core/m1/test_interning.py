"""Structural interning (M1 Acceptance Contract).

Identical structure -> identical NodeId; any structural difference -> distinct NodeId. Float edge
cases (0.0 / -0.0 / NaN) are handled by a deterministic total order: NaN interns to itself,
0.0 and -0.0 are distinct syntactic literals (canonicalizing them is M4's job, not M1's).
"""

from __future__ import annotations

import math

import graphed_core as gc


def test_identical_ops_intern_to_one_node() -> None:
    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "f.root"})
    a = s.add_op("pt", [src])
    b = s.add_op("pt", [src])
    assert a == b
    assert s.node_count() == 2  # source + the single deduped op


def test_distinct_structure_distinct_ids() -> None:
    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "f.root"})
    a = s.add_op("pt", [src])
    b = s.add_op("eta", [src])  # different name
    c = s.add_op("pt", [src], {"scale": 2})  # different params
    assert len({a, b, c}) == 3
    assert s.node_count() == 4


def test_input_order_matters() -> None:
    s = gc.GraphStore()
    x = s.add_source("x")
    y = s.add_source("y")
    a = s.add_op("add", [x, y])
    b = s.add_op("add", [y, x])
    assert a != b


def test_param_value_and_type_matter() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    a = s.add_op("cut", [src], {"thr": 30})
    b = s.add_op("cut", [src], {"thr": 30})
    c = s.add_op("cut", [src], {"thr": 31})
    d = s.add_op("cut", [src], {"thr": "30"})  # str vs int distinct
    assert a == b
    assert a != c
    assert a != d


def test_float_zero_signed_are_distinct() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    pos = s.add_op("k", [src], {"v": 0.0})
    neg = s.add_op("k", [src], {"v": -0.0})
    assert pos != neg


def test_nan_interns_to_itself() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    a = s.add_op("k", [src], {"v": float("nan")})
    b = s.add_op("k", [src], {"v": float("nan")})
    c = s.add_op("k", [src], {"v": math.nan})
    assert a == b == c  # all NaNs canonicalize to one structural key


def test_nan_distinct_from_zero() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    nan = s.add_op("k", [src], {"v": float("nan")})
    zero = s.add_op("k", [src], {"v": 0.0})
    assert nan != zero


def test_float_and_int_keys_distinct() -> None:
    s = gc.GraphStore()
    src = s.add_source("e")
    fi = s.add_op("k", [src], {"v": 1.0})
    ii = s.add_op("k", [src], {"v": 1})
    assert fi != ii
