"""Fixtures for the m71a numpy suite: a numpy-only session with two float32 vectors and a 2-D array.

Every m71a-new surface (the `output_type=` keyword) is reached inside test bodies only, so this tree
collects on a base without it and fails at run time.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from graphed import GraphedTypeError, Session
from graphed.numpy import NumpyBackend, from_array, from_record

X = np.array([0.5, 1.5], dtype=np.float32)
Y = np.array([2.0, 0.5], dtype=np.float32)
M2 = np.arange(6, dtype=np.float32).reshape(2, 3)
RA = np.array([0.5, 2.0])
RB = np.array([1, 2])

#: every numpy numerical dtype (``np.typecodes`` "AllInteger" + "AllFloat" + bool, fixed-width only)
NUMERIC = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


def f(a: Any) -> Any:
    return a > 1


def recorded() -> tuple[Session, Any, Any, Any]:
    """A numpy session with `x`, `y` (float32 vectors) and the 2-D `m2`."""
    sn = Session(NumpyBackend())
    return sn, from_array(sn, "x", X), from_array(sn, "y", Y), from_array(sn, "m2", M2)


def record_of(sn: Session) -> Any:
    return from_record(sn, "r", a=RA, b=RB)


def describe(session: Session, array: Any) -> str:
    return str(session.form(array).describe())


def params_of(session: Session, array: Any) -> dict[str, Any]:
    """The record-time store params behind `array` (the house route)."""
    node = next(n for n in session._store.nodes() if n["id"] == array.node_id)
    return dict(node["params"])


def refusal(call: Callable[[], object]) -> GraphedTypeError:
    try:
        call()
    except GraphedTypeError as exc:
        return exc
    raise AssertionError("expected a GraphedTypeError")


def here() -> int:
    """The caller's NEXT line: the line a one-line refused call sits on."""
    return sys._getframe(1).f_lineno + 1
