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
# M11 (dask.array parity P0.2): the FULL single-output ufunc tier. Ufunc application is COMMON to
# both backend idioms (numpy arrays and awkward arrays both accept ufuncs), so it lives here;
# numpy aliases (degrees/rad2deg, divide/true_divide, ...) map to one canonical name so
# hash-consing dedups them.
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

    # ---- shared backend-proxy infrastructure (M11/M12, dask.array parity P0/P1) -
    # graphed.Array carries only what is COMMON to the backend idioms. Idiomatic surfaces live on
    # the backend's proxy subclass (graphed_numpy.NumpyArray: .shape/.sum()/__array_function__) or
    # in its function namespace (graphed_awkward.gak — awkward arrays never grow methods). These
    # protected helpers are the infrastructure those surfaces are built from.
    def _form_meta(self, name: str) -> Any:
        """Answer array metadata from the node's form when the backend models it; otherwise fall
        back to recording a field op, preserving the M3 attribute-access semantics for backends
        whose records may genuinely contain a column of that name."""
        form = self._session.form(self)
        if hasattr(form, name):
            return getattr(form, name)
        return self._session.record_op("field", [self], {"field": name})

    def _norm_axis(self, axis: int | None) -> int | None:
        """Resolve a (possibly negative) axis against the form's ndim, when the backend models it."""
        if axis is None:
            return None
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"axis must be an int or None, got {axis!r}")
        if axis < 0:
            ndim = getattr(self._session.form(self), "ndim", None)
            if not isinstance(ndim, int):
                raise TypeError("a negative axis requires a backend form exposing ndim")
            axis += ndim
            if axis < 0:
                raise TypeError(f"axis out of range for a {ndim}-dimensional array")
        return axis

    def _reduction(
        self, kind: str, axis: int | None = None, *, keepdims: bool = False, ddof: int = 0
    ) -> Array:
        """Record one reduction with THE structural rule of M12 (dask.array parity P1.4):
        reducing over the partitioned axis (``axis`` None or 0) is a stage boundary executed by
        the M7 tree reduction; an inner axis is partition-local and fusible."""
        axis = self._norm_axis(axis)
        params: dict[str, ParamValue] = {}
        if axis is not None:
            params["axis"] = axis
        if keepdims:
            params["keepdims"] = True
        if ddof:  # default ddof=0 records nothing, so std(ddof=0) interns with std()
            params["ddof"] = ddof
        return self._session.record_op(kind, [self], params, reduction=axis is None or axis == 0)

    def _scan(self, kind: str, axis: int | None = None) -> Array:
        """Record one cumulative scan — always partition-local, always fusible."""
        axis = self._norm_axis(axis)
        params: dict[str, ParamValue] = {}
        if axis is not None:
            params["axis"] = axis
        return self._session.record_op(kind, [self], params)

    # ---- structural access -----------------------------------------------------
    def __getattr__(self, name: str) -> Array:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._session.record_op("field", [self], {"field": name})

    def __iter__(self) -> Any:
        # int __getitem__ (M13) would otherwise make Array INFINITELY iterable through Python's
        # legacy iteration protocol (a[0], a[1], ... never raises IndexError on a deferred graph)
        raise TypeError(
            "deferred graphed arrays are not iterable (unknown partitioned length); materialize first"
        )

    def __getitem__(self, key: object) -> Array:
        if isinstance(key, Array):
            return self._session.record_op("getitem", [self, key])
        if isinstance(key, str):
            return self._session.record_op("field", [self], {"field": key})
        # M13: slices and integer indexing are common to both backend idioms. Both consume or
        # restructure the partitioned axis, so they record BOUNDARY reduction nodes (M12 rule);
        # only the fields the user gave are recorded, so equal slices intern.
        if isinstance(key, slice):
            params: dict[str, ParamValue] = {}
            for name, value in (("start", key.start), ("stop", key.stop), ("step", key.step)):
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"slice fields must be ints, got {value!r}")
                params[name] = value
            return self._session.record_op("slice", [self], params, reduction=True)
        if not isinstance(key, bool) and isinstance(key, int):
            return self._session.record_op("index", [self], {"i": key}, reduction=True)
        raise TypeError(
            f"unsupported index {key!r}; use an Array mask/index, a slice, an int, or a field name"
        )

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


def apply(fn: Callable[..., object], *arrays: Array, name: str | None = None) -> Array:
    """Record ``fn`` over several deferred arrays as ONE multi-input External node (M14, parity
    P3.8 — the blockwise/map_blocks analogue). A function over arrays, so it is idiom-neutral
    (awkward style); the numpy-specific signature-aware form is ``graphed_numpy.apply_gufunc``.

    The node carries the backend's ``PayloadDescriptor``: the opaque callable stays a flagged
    preservation risk (plan A.3.1). With one array this IS ``Array.map`` (interns with it)."""
    if not arrays or not all(isinstance(a, Array) for a in arrays):
        raise TypeError("apply needs at least one deferred Array operand")
    session = arrays[0].session
    if any(a.session is not session for a in arrays):
        raise TypeError("apply operands must come from one Session")
    fn_name: str = name or str(getattr(fn, "__name__", "lambda"))
    return session.record_external("map", fn, list(arrays), {"fn": fn_name})
