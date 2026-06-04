"""The deferred Array proxy (M2 + the awkward op surface added in M3).

An `Array` is a typed handle to one interned node. Operators, field access, indexing, and numpy
ufuncs all record new nodes; repeated identical sub-expressions intern to the same node. The Array
holds no backend data — only a session + node id. Op names recorded here are *canonical* (backend
dispatch keys); they are backend-agnostic strings, so graphed stays free of numpy/awkward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, SupportsFloat

from .backend import ParamValue

if TYPE_CHECKING:
    from .session import Session

# numpy ufunc name -> canonical op name (so np.subtract(a, b) and a - b record the same op).
_UFUNC_TO_OP: dict[str, str] = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "true_divide": "div",
    "divide": "div",
    "greater": "gt",
    "less": "lt",
    "greater_equal": "ge",
    "less_equal": "le",
    "equal": "eq",
    "not_equal": "ne",
    "absolute": "abs",
    "negative": "neg",
    "sqrt": "sqrt",
    "cos": "cos",
    "sin": "sin",
    "cosh": "cosh",
    "sinh": "sinh",
    "hypot": "hypot",
    "maximum": "maximum",
    "minimum": "minimum",
    "power": "power",
}


def _as_param(value: object) -> ParamValue:
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, SupportsFloat):  # coerces numpy scalars without importing numpy
        return float(value)
    raise TypeError(f"unsupported scalar operand {value!r}")


class Array:
    __slots__ = ("_node_id", "_session")

    # let numpy defer to us for ufuncs (np.cos(array), np.hypot(a, b), ...)
    __array_priority__ = 1000.0

    def __init__(self, session: Session, node_id: int) -> None:
        self._session = session
        self._node_id = node_id

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def session(self) -> Session:
        return self._session

    # ---- elementwise recording -------------------------------------------------
    def _binary(self, op: str, other: object, *, reflected: bool = False) -> Array:
        if isinstance(other, Array):
            inputs = [other, self] if reflected else [self, other]
            return self._session.record_op(op, inputs)
        params: dict[str, ParamValue] = {"scalar": _as_param(other), "side": "l" if reflected else "r"}
        return self._session.record_op(op, [self], params)

    def _unary(self, op: str) -> Array:
        return self._session.record_op(op, [self])

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: object, **kwargs: object) -> Any:
        if method != "__call__" or kwargs:
            return NotImplemented
        op = _UFUNC_TO_OP.get(getattr(ufunc, "__name__", ""))
        if op is None:
            return NotImplemented
        if len(inputs) == 1:
            return self._unary(op)
        if len(inputs) == 2:
            left, right = inputs
            if left is self:
                return self._binary(op, right)
            return self._binary(op, left, reflected=True)
        return NotImplemented

    # arithmetic
    def __add__(self, other: object) -> Array:
        return self._binary("add", other)

    def __radd__(self, other: object) -> Array:
        return self._binary("add", other, reflected=True)

    def __sub__(self, other: object) -> Array:
        return self._binary("sub", other)

    def __rsub__(self, other: object) -> Array:
        return self._binary("sub", other, reflected=True)

    def __mul__(self, other: object) -> Array:
        return self._binary("mul", other)

    def __rmul__(self, other: object) -> Array:
        return self._binary("mul", other, reflected=True)

    def __truediv__(self, other: object) -> Array:
        return self._binary("div", other)

    def __rtruediv__(self, other: object) -> Array:
        return self._binary("div", other, reflected=True)

    def __pow__(self, other: object) -> Array:
        return self._binary("power", other)

    def __mod__(self, other: object) -> Array:
        return self._binary("mod", other)

    def __rmod__(self, other: object) -> Array:
        return self._binary("mod", other, reflected=True)

    def __abs__(self) -> Array:
        return self._unary("abs")

    def __neg__(self) -> Array:
        return self._unary("neg")

    # comparisons (deferred -> Array, not bool)
    def __gt__(self, other: object) -> Array:
        return self._binary("gt", other)

    def __lt__(self, other: object) -> Array:
        return self._binary("lt", other)

    def __ge__(self, other: object) -> Array:
        return self._binary("ge", other)

    def __le__(self, other: object) -> Array:
        return self._binary("le", other)

    def __eq__(self, other: object) -> Array:  # type: ignore[override]
        return self._binary("eq", other)

    def __ne__(self, other: object) -> Array:  # type: ignore[override]
        return self._binary("ne", other)

    __hash__ = None  # type: ignore[assignment]  # deferred __eq__ makes Array unhashable

    # boolean combination of masks
    def __and__(self, other: object) -> Array:
        return self._binary("and", other)

    def __or__(self, other: object) -> Array:
        return self._binary("or", other)

    def __invert__(self) -> Array:
        return self._unary("invert")

    # ---- structural access -----------------------------------------------------
    def __getattr__(self, name: str) -> Array:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._session.record_op("field", [self], {"field": name})

    def __getitem__(self, key: object) -> Array:
        if isinstance(key, Array):
            return self._session.record_op("getitem", [self, key])
        if isinstance(key, str):
            return self._session.record_op("field", [self], {"field": key})
        raise TypeError(f"unsupported index {key!r}; use an Array mask/index or a field name")

    # ---- M2 methods (kept) -----------------------------------------------------
    def filter(self, mask: Array) -> Array:
        return self._session.record_op("filter", [self, mask])

    def map(self, fn: Callable[[object], object], *, name: str | None = None) -> Array:
        fn_name: str = name or str(getattr(fn, "__name__", "lambda"))
        return self._session.record_external("map", fn, [self], {"fn": fn_name})

    def reduce(self, kind: str = "sum") -> Array:
        return self._session.record_op(kind, [self], reduction=True)

    def __repr__(self) -> str:
        return f"Array(node_id={self._node_id})"
