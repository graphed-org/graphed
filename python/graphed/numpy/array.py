"""``NumpyArray``: the numpy-idiomatic deferred-array proxy (M11, dask.array parity P0).

The shared ``graphed.Array`` is backend-idiom-neutral — it carries operators, ufunc dispatch, and
protected infrastructure, but no numpy-specific surface (awkward's idiom applies operations as
functions over arrays, never as member functions). This subclass COMPLETES the numpy idiom for the
numpy backend: metadata properties (``.shape/.dtype/.ndim`` answered from the form) and the
``__array_function__`` protocol so numpy API calls (``np.sum(a)``, …) record canonical ops.
``NumpyBackend.array_type`` hands this class to the Session, so every recorded node comes back as
a ``NumpyArray``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graphed import Array

Handler = Callable[["NumpyArray", tuple[object, ...], dict[str, object]], Array]


class NumpyArray(Array):
    """A deferred array with numpy's calling idiom (methods, properties, numpy API dispatch)."""

    __slots__ = ()

    # ---- array metadata (answered from the form; never recorded) ----------------
    @property
    def shape(self) -> Any:
        return self._form_meta("shape")

    @property
    def dtype(self) -> Any:
        return self._form_meta("dtype")

    @property
    def ndim(self) -> Any:
        return self._form_meta("ndim")

    # ---- reductions & scans (M12, dask.array parity P1.4) ------------------------
    # numpy's METHOD idiom over the base class's _reduction/_scan infrastructure (the structural
    # rule — axis None/0 is a boundary reduction, an inner axis is fusible — lives there).
    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("sum", axis, keepdims=keepdims)

    def prod(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("prod", axis, keepdims=keepdims)

    def mean(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("mean", axis, keepdims=keepdims)

    def std(self, axis: int | None = None, *, keepdims: bool = False, ddof: int = 0) -> Array:
        return self._reduction("std", axis, keepdims=keepdims, ddof=ddof)

    def var(self, axis: int | None = None, *, keepdims: bool = False, ddof: int = 0) -> Array:
        return self._reduction("var", axis, keepdims=keepdims, ddof=ddof)

    def min(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("min", axis, keepdims=keepdims)

    def max(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("max", axis, keepdims=keepdims)

    def any(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("any", axis, keepdims=keepdims)

    def all(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("all", axis, keepdims=keepdims)

    def argmin(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("argmin", axis, keepdims=keepdims)

    def argmax(self, axis: int | None = None, *, keepdims: bool = False) -> Array:
        return self._reduction("argmax", axis, keepdims=keepdims)

    def cumsum(self, axis: int | None = None) -> Array:
        return self._scan("cumsum", axis)

    def cumprod(self, axis: int | None = None) -> Array:
        return self._scan("cumprod", axis)

    # ---- numpy API dispatch ------------------------------------------------------
    def __array_function__(
        self, func: Any, types: object, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> Any:
        handler = _NUMPY_FUNCTIONS.get(getattr(func, "__name__", ""))
        if handler is None:
            return NotImplemented  # numpy raises TypeError naming the unsupported function
        return handler(self, args, kwargs)


def _take_axis(
    kind: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[int | None, dict[str, object]]:
    """Normalize the (positional-or-keyword) axis argument of a numpy reduction/scan call."""
    kw = dict(kwargs)
    rest = args[1:]
    if rest:
        if len(rest) > 1 or "axis" in kw:
            raise TypeError(f"graphed np.{kind} takes at most one positional argument after the array")
        kw["axis"] = rest[0]
    axis = kw.pop("axis", None)
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, int)):
        raise TypeError(f"graphed np.{kind}: axis must be an int or None, got {axis!r}")
    return axis, kw


def _make_reduction(kind: str, *, allow_ddof: bool = False) -> Handler:
    def handler(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Array:
        if not args or args[0] is not arr:
            raise TypeError(f"graphed np.{kind}: the first argument must be the deferred array")
        axis, kw = _take_axis(kind, args, kwargs)
        keepdims = bool(kw.pop("keepdims", False))
        ddof_obj = kw.pop("ddof", 0) if allow_ddof else 0
        if kw:
            raise TypeError(f"graphed np.{kind} does not support arguments {sorted(kw)}")
        ddof = ddof_obj if isinstance(ddof_obj, int) else 0
        return arr._reduction(kind, axis, keepdims=keepdims, ddof=ddof)

    return handler


def _make_scan(kind: str) -> Handler:
    def handler(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Array:
        if not args or args[0] is not arr:
            raise TypeError(f"graphed np.{kind}: the first argument must be the deferred array")
        axis, kw = _take_axis(kind, args, kwargs)
        if kw:
            raise TypeError(f"graphed np.{kind} does not support arguments {sorted(kw)}")
        return arr._scan(kind, axis)

    return handler


# numpy API function name -> recorder. M11 wired the protocol + sum; M12 adds the axis-aware
# reduction/scan tier; M13 extends with manipulation routines without re-touching the protocol.
_NUMPY_FUNCTIONS: dict[str, Handler] = {
    name: _make_reduction(kind)
    for name, kind in [
        ("sum", "sum"),
        ("prod", "prod"),
        ("mean", "mean"),
        ("min", "min"),
        ("amin", "min"),
        ("max", "max"),
        ("amax", "max"),
        ("any", "any"),
        ("all", "all"),
        ("argmin", "argmin"),
        ("argmax", "argmax"),
        ("nansum", "nansum"),
        ("nanprod", "nanprod"),
        ("nanmean", "nanmean"),
        ("nanmin", "nanmin"),
        ("nanmax", "nanmax"),
        ("nanargmin", "nanargmin"),
        ("nanargmax", "nanargmax"),
    ]
}
_NUMPY_FUNCTIONS.update(
    {name: _make_reduction(name, allow_ddof=True) for name in ("std", "var", "nanstd", "nanvar")}
)
_NUMPY_FUNCTIONS.update({name: _make_scan(name) for name in ("cumsum", "cumprod", "nancumsum", "nancumprod")})
