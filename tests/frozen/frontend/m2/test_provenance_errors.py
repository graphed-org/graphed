"""Provenance stub + ill-typed ops raise at the user's source line (M2)."""

from __future__ import annotations

import sys

from backends import ListBackend, from_list

from graphed import GraphedTypeError, Session


def test_provenance_points_at_build_line() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [4, 5, 6])
    line = sys._getframe().f_lineno + 1
    c = a + b
    assert s.provenance(c).filename == __file__
    assert s.provenance(c).lineno == line


def test_ill_typed_op_raises_at_user_line() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    mask = from_list(s, "m", [1, 0, 1])  # ints, not bool -> filter is ill-typed
    captured: GraphedTypeError | None = None
    try:
        call_line = sys._getframe().f_lineno + 1
        a.filter(mask)
    except GraphedTypeError as exc:
        captured = exc
    assert captured is not None, "expected GraphedTypeError"
    assert captured.op == "filter"
    assert captured.provenance.filename == __file__
    assert captured.provenance.lineno == call_line


def test_ill_typed_arith_raises() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    flags = from_list(s, "f", [True, False, True])  # bool, not numeric
    try:
        a + flags
    except GraphedTypeError as exc:
        assert exc.op == "add"
        assert exc.provenance.filename == __file__
    else:
        raise AssertionError("expected GraphedTypeError")
