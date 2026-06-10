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
# M11 (dask.array parity P0.2): the FULL single-output ufunc tier. numpy aliases (degrees/rad2deg,
# divide/true_divide, ...) map to one canonical name so hash-consing dedups them.
_UFUNC_TO_OP: dict[str, str] = {
    # arithmetic
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "true_divide": "div",
    "divide": "div",
    "floor_divide": "floordiv",
    "remainder": "mod",
    "power": "power",
    "float_power": "float_power",
    "fmod": "fmod",
    # comparisons
    "greater": "gt",
    "less": "lt",
    "greater_equal": "ge",
    "less_equal": "le",
    "equal": "eq",
    "not_equal": "ne",
    # signs / rounding
    "absolute": "abs",
    "fabs": "fabs",
    "negative": "neg",
    "positive": "pos",
    "sign": "sign",
    "signbit": "signbit",
    "copysign": "copysign",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "rint": "rint",
    # exponentials / logs / powers
    "exp": "exp",
    "exp2": "exp2",
    "expm1": "expm1",
    "log": "log",
    "log1p": "log1p",
    "log2": "log2",
    "log10": "log10",
    "logaddexp": "logaddexp",
    "logaddexp2": "logaddexp2",
    "sqrt": "sqrt",
    "cbrt": "cbrt",
    "square": "square",
    "reciprocal": "reciprocal",
    # trig / hyperbolic
    "cos": "cos",
    "sin": "sin",
    "tan": "tan",
    "cosh": "cosh",
    "sinh": "sinh",
    "tanh": "tanh",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arctan2": "arctan2",
    "arcsinh": "arcsinh",
    "arccosh": "arccosh",
    "arctanh": "arctanh",
    "hypot": "hypot",
    "deg2rad": "deg2rad",
    "rad2deg": "rad2deg",
    "degrees": "rad2deg",
    "radians": "deg2rad",
    # extrema
    "maximum": "maximum",
    "minimum": "minimum",
    "fmax": "fmax",
    "fmin": "fmin",
    # floating-point inspection / manipulation
    "isnan": "isnan",
    "isinf": "isinf",
    "isfinite": "isfinite",
    "nextafter": "nextafter",
    "spacing": "spacing",
    "ldexp": "ldexp",
    "heaviside": "heaviside",
    "conjugate": "conj",
    # integer / bitwise
    "gcd": "gcd",
    "lcm": "lcm",
    "bitwise_and": "and",
    "bitwise_or": "or",
    "bitwise_xor": "xor",
    "invert": "invert",
    "left_shift": "lshift",
    "right_shift": "rshift",
    # logical
    "logical_and": "logical_and",
    "logical_or": "logical_or",
    "logical_xor": "logical_xor",
    "logical_not": "logical_not",
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

    def __rpow__(self, other: object) -> Array:
        return self._binary("power", other, reflected=True)

    def __floordiv__(self, other: object) -> Array:
        return self._binary("floordiv", other)

    def __rfloordiv__(self, other: object) -> Array:
        return self._binary("floordiv", other, reflected=True)

    def __mod__(self, other: object) -> Array:
        return self._binary("mod", other)

    def __rmod__(self, other: object) -> Array:
        return self._binary("mod", other, reflected=True)

    def __abs__(self) -> Array:
        return self._unary("abs")

    def __neg__(self) -> Array:
        return self._unary("neg")

    def __pos__(self) -> Array:
        return self._unary("pos")

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

    # boolean / bitwise combination
    def __and__(self, other: object) -> Array:
        return self._binary("and", other)

    def __rand__(self, other: object) -> Array:
        return self._binary("and", other, reflected=True)

    def __or__(self, other: object) -> Array:
        return self._binary("or", other)

    def __ror__(self, other: object) -> Array:
        return self._binary("or", other, reflected=True)

    def __xor__(self, other: object) -> Array:
        return self._binary("xor", other)

    def __rxor__(self, other: object) -> Array:
        return self._binary("xor", other, reflected=True)

    def __lshift__(self, other: object) -> Array:
        return self._binary("lshift", other)

    def __rlshift__(self, other: object) -> Array:
        return self._binary("lshift", other, reflected=True)

    def __rshift__(self, other: object) -> Array:
        return self._binary("rshift", other)

    def __rrshift__(self, other: object) -> Array:
        return self._binary("rshift", other, reflected=True)

    def __invert__(self) -> Array:
        return self._unary("invert")

    # ---- array metadata (M11, dask.array parity P0.1) ---------------------------
    def _form_meta(self, name: str) -> Any:
        """Answer from the node's form when the backend models the metadata; otherwise fall back
        to recording a field op, preserving the M3 attribute-access semantics for backends whose
        records may genuinely contain a column of that name."""
        form = self._session.form(self)
        if hasattr(form, name):
            return getattr(form, name)
        return self._session.record_op("field", [self], {"field": name})

    @property
    def shape(self) -> Any:
        return self._form_meta("shape")

    @property
    def dtype(self) -> Any:
        return self._form_meta("dtype")

    @property
    def ndim(self) -> Any:
        return self._form_meta("ndim")

    # ---- numpy API dispatch (M11, dask.array parity P0.3) -----------------------
    def __array_function__(
        self, func: Any, types: object, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> Any:
        handler = _ARRAY_FUNCTIONS.get(getattr(func, "__name__", ""))
        if handler is None:
            return NotImplemented  # numpy raises TypeError naming the unsupported function
        return handler(self, args, kwargs)

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


def _fn_sum(arr: Array, args: tuple[object, ...], kwargs: dict[str, object]) -> Array:
    if len(args) != 1 or args[0] is not arr or kwargs:
        raise TypeError("graphed records np.sum(array) with no extra arguments (axis-aware reductions: M12)")
    return arr.reduce("sum")


# numpy API function name -> recorder. M11 wires the protocol + whole-array sum; later milestones
# extend this table (axis-aware reductions M12; concatenate/where/take M13) without re-touching it.
_ARRAY_FUNCTIONS: dict[str, Callable[[Array, tuple[object, ...], dict[str, object]], Array]] = {
    "sum": _fn_sum,
}
