"""The deferred Array proxy (plan M2).

An `Array` is a typed handle to one interned node. Operators record new nodes into the session's
store; repeated identical sub-expressions intern to the same node (zero new nodes). The Array holds
no backend data — only a session + node id.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Session


class Array:
    __slots__ = ("_node_id", "_session")

    def __init__(self, session: Session, node_id: int) -> None:
        self._session = session
        self._node_id = node_id

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def session(self) -> Session:
        return self._session

    def _binop(self, op: str, other: Array) -> Array:
        return self._session.record_op(op, [self, other])

    def __add__(self, other: Array) -> Array:
        return self._binop("add", other)

    def __sub__(self, other: Array) -> Array:
        return self._binop("sub", other)

    def __mul__(self, other: Array) -> Array:
        return self._binop("mul", other)

    def __truediv__(self, other: Array) -> Array:
        return self._binop("div", other)

    def filter(self, mask: Array) -> Array:
        """Keep elements where `mask` is true."""
        return self._session.record_op("filter", [self, mask])

    def map(self, fn: Callable[[object], object], *, name: str | None = None) -> Array:
        """Apply an opaque Python callable — recorded as an External node (preservation risk)."""
        fn_name: str = name or str(getattr(fn, "__name__", "lambda"))
        return self._session.record_external("map", fn, [self], {"fn": fn_name})

    def reduce(self, kind: str = "sum") -> Array:
        """A boundary reduction over the array."""
        return self._session.record_op(kind, [self], reduction=True)

    def __repr__(self) -> str:
        return f"Array(node_id={self._node_id})"
