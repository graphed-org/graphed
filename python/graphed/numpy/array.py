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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from graphed import Array
from graphed.backend import ParamValue

Handler = Callable[["NumpyArray", tuple[object, ...], dict[str, object]], Any]


def _encode_dims(dims: Sequence[int]) -> str:
    out: list[int] = []
    for d in dims:
        if isinstance(d, bool) or not isinstance(d, int):
            raise TypeError(f"dimensions must be ints, got {d!r}")
        out.append(d)
    return ",".join(str(d) for d in out)


def _encode_subscript(key: tuple[object, ...]) -> str:
    """Canonical injective spec for an inner (partition-local) tuple subscript.

    The first element must be the FULL slice ``:`` — anything indexing the partitioned axis 0
    inside a tuple is refused in the axis-0-partitioned MVP (Phase 2: N-D chunking)."""
    if not key or key[0] != slice(None, None, None):
        raise TypeError("tuple subscripts must keep the partitioned axis 0 whole: a[:, inner...]")
    parts: list[str] = []
    for elem in key:
        if isinstance(elem, slice):
            bits = []
            for v in (elem.start, elem.stop, elem.step):
                if v is None:
                    bits.append("")
                elif isinstance(v, bool) or not isinstance(v, int):
                    raise TypeError(f"slice fields must be ints, got {v!r}")
                else:
                    bits.append(str(v))
            parts.append(":".join(bits))
        elif not isinstance(elem, bool) and isinstance(elem, int):
            parts.append(str(elem))
        else:
            raise TypeError(f"unsupported tuple-subscript element {elem!r}")
    return ",".join(parts)


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

    # ---- indexing (M13, dask.array parity P2.6) ----------------------------------
    def __getitem__(self, key: object) -> Array:
        if isinstance(key, tuple):
            # the numpy tuple-subscript idiom: partition-local inner indexing, fusible
            return self._session.record_op("subscript", [self], {"spec": _encode_subscript(key)})
        return super().__getitem__(key)

    # ---- manipulation (M13, dask.array parity P2.6) ------------------------------
    def reshape(self, *shape: int | tuple[int, ...]) -> Array:
        dims = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return self._session.record_op("reshape", [self], {"shape": _encode_dims(dims)})  # type: ignore[arg-type]

    def ravel(self) -> Array:
        return self._session.record_op("ravel", [self])

    def squeeze(self, axis: int | None = None) -> Array:
        if axis is None:
            raise TypeError(
                "squeeze needs an explicit inner axis: squeezing ALL size-1 dims could silently "
                "eat a length-1 partition of the partitioned axis"
            )
        return self._session.record_op("squeeze", [self], {"axis": self._subaxis(axis)})

    def transpose(self, *axes: int | tuple[int, ...]) -> Array:
        flat = axes[0] if len(axes) == 1 and isinstance(axes[0], tuple) else axes
        params: dict[str, ParamValue] = {"axes": _encode_dims(flat)} if flat else {}  # type: ignore[arg-type]
        return self._session.record_op("transpose", [self], params)

    @property
    def T(self) -> Array:  # numpy parity name
        return self.transpose()

    def swapaxes(self, a1: int, a2: int) -> Array:
        return self._session.record_op("swapaxes", [self], {"a1": self._subaxis(a1), "a2": self._subaxis(a2)})

    def astype(self, dtype: object) -> Array:
        return self._session.record_op("astype", [self], {"dtype": np.dtype(dtype).str})

    def clip(self, lo: float | None = None, hi: float | None = None) -> Array:
        if lo is None and hi is None:
            raise TypeError("clip needs at least one bound")
        params: dict[str, ParamValue] = {}
        if lo is not None:
            params["lo"] = float(lo)
        if hi is not None:
            params["hi"] = float(hi)
        return self._session.record_op("clip", [self], params)

    def round(self, decimals: int = 0) -> Array:
        params: dict[str, ParamValue] = {"decimals": int(decimals)} if decimals else {}
        return self._session.record_op("round", [self], params)

    def take(self, indices: Array, axis: int | None = None) -> Array:
        axis = self._norm_axis(axis)
        params: dict[str, ParamValue] = {} if axis is None else {"axis": axis}
        # gathering by global index along the partitioned axis crosses partitions: boundary
        return self._session.record_op("take", [self, indices], params, reduction=axis is None or axis == 0)

    def _subaxis(self, axis: int) -> int:
        """Normalize an axis that must end up addressing an INNER (non-partitioned) dimension."""
        normalized = self._norm_axis(axis)
        assert normalized is not None
        return normalized

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


# ---- M13: manipulation + combination handlers (dask.array parity P2) ------------------------------
def _one_array(kind: str, args: tuple[object, ...]) -> NumpyArray:
    if not args or not isinstance(args[0], NumpyArray):
        raise TypeError(f"graphed np.{kind}: the first argument must be the deferred array")
    return args[0]


def _two_arrays(kind: str, args: tuple[object, ...]) -> tuple[NumpyArray, NumpyArray]:
    if len(args) != 2 or not isinstance(args[0], NumpyArray) or not isinstance(args[1], NumpyArray):
        raise TypeError(f"graphed np.{kind} takes two deferred arrays")
    return args[0], args[1]


def _fn_reshape(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("reshape", args)
    if len(args) != 2 or kwargs:
        raise TypeError("graphed np.reshape takes (array, shape)")
    shape = args[1] if isinstance(args[1], tuple) else (args[1],)
    return a.reshape(shape)


def _fn_ravel(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    if len(args) != 1 or kwargs:
        raise TypeError("graphed np.ravel takes (array,)")
    return _one_array("ravel", args).ravel()


def _fn_squeeze(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("squeeze", args)
    axis = args[1] if len(args) > 1 else kwargs.pop("axis", None)
    if kwargs or len(args) > 2:
        raise TypeError("graphed np.squeeze takes (array, axis)")
    return a.squeeze(axis)  # type: ignore[arg-type]


def _fn_transpose(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("transpose", args)
    if kwargs or len(args) > 2:
        raise TypeError("graphed np.transpose takes (array, axes)")
    if len(args) == 2 and args[1] is not None:
        axes = args[1]
        if not isinstance(axes, list | tuple):
            raise TypeError("graphed np.transpose: axes must be a tuple of ints")
        return a.transpose(tuple(axes))
    return a.transpose()


def _fn_swapaxes(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("swapaxes", args)
    if len(args) != 3 or kwargs:
        raise TypeError("graphed np.swapaxes takes (array, axis1, axis2)")
    return a.swapaxes(int(args[1]), int(args[2]))  # type: ignore[call-overload]


def _fn_expand_dims(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("expand_dims", args)
    axis_obj = args[1] if len(args) > 1 else kwargs.pop("axis", None)
    if kwargs or len(args) > 2 or isinstance(axis_obj, bool) or not isinstance(axis_obj, int):
        raise TypeError("graphed np.expand_dims takes (array, axis:int)")
    axis = axis_obj + a.ndim + 1 if axis_obj < 0 else axis_obj
    return a.session.record_op("expand_dims", [a], {"axis": axis})


def _fn_clip(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("clip", args)
    kw = dict(kwargs)
    lo: object = args[1] if len(args) > 1 else kw.pop("a_min", kw.pop("min", None))
    hi: object = args[2] if len(args) > 2 else kw.pop("a_max", kw.pop("max", None))
    if kw or len(args) > 3:
        raise TypeError("graphed np.clip takes (array, min, max)")
    return a.clip(None if lo is None else float(lo), None if hi is None else float(hi))  # type: ignore[arg-type]


def _fn_round(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("round", args)
    decimals = args[1] if len(args) > 1 else kwargs.pop("decimals", 0)
    if kwargs or len(args) > 2:
        raise TypeError("graphed np.round takes (array, decimals)")
    return a.round(int(decimals))  # type: ignore[call-overload]


def _fn_take(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("take", args)
    indices = args[1] if len(args) > 1 else None
    axis = kwargs.pop("axis", None)
    if kwargs or len(args) > 2 or not isinstance(indices, NumpyArray):
        raise TypeError("graphed np.take takes (array, deferred indices, axis=)")
    return a.take(indices, None if axis is None else int(axis))  # type: ignore[call-overload]


def _fn_astype(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("astype", args)
    if len(args) != 2 or kwargs:
        raise TypeError("graphed np.astype takes (array, dtype)")
    return a.astype(args[1])


def _fn_where(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    if kwargs or len(args) != 3:
        raise TypeError("graphed np.where needs (cond, x, y)")
    cond, x, y = args
    if not isinstance(cond, NumpyArray):
        raise TypeError("graphed np.where: the condition must be a deferred array")
    inputs: list[Array] = [cond]
    params: dict[str, ParamValue] = {}
    if isinstance(x, NumpyArray):
        inputs.append(x)
    else:
        params["x_scalar"] = float(x)  # type: ignore[arg-type]
    if isinstance(y, NumpyArray):
        inputs.append(y)
    else:
        params["y_scalar"] = float(y)  # type: ignore[arg-type]
    return cond.session.record_op("where", inputs, params)


def _record_concat(arrays: list[NumpyArray], axis: int) -> Array:
    params: dict[str, ParamValue] = {"axis": axis} if axis else {}
    # axis 0 RESTRUCTURES the partition set: boundary; inner axes zip partitions: fusible
    return arrays[0].session.record_op("concatenate", arrays, params, reduction=axis == 0)


def _seq_of_arrays(kind: str, args: tuple[object, ...]) -> list[NumpyArray]:
    if not args or not isinstance(args[0], list | tuple):
        raise TypeError(f"graphed np.{kind} needs a sequence of deferred arrays")
    arrays = list(args[0])
    if not arrays or not all(isinstance(a, NumpyArray) for a in arrays):
        raise TypeError(f"graphed np.{kind} needs a sequence of deferred arrays")
    return arrays


def _fn_concatenate(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    arrays = _seq_of_arrays("concatenate", args)
    axis = args[1] if len(args) > 1 else kwargs.pop("axis", 0)
    if kwargs or len(args) > 2:
        raise TypeError("graphed np.concatenate takes (arrays, axis)")
    return _record_concat(arrays, int(axis))  # type: ignore[call-overload]


def _fn_hstack(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    arrays = _seq_of_arrays("hstack", args)
    if kwargs or len(args) > 1:
        raise TypeError("graphed np.hstack takes (arrays,)")
    return _record_concat(arrays, 0 if arrays[0].ndim == 1 else 1)


def _fn_stack_refused(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    raise TypeError(
        "np.stack/np.vstack would create an inner unknown-length dimension, which the "
        "axis-0-partitioned MVP cannot represent (Phase 2: N-D chunking)"
    )


def _fn_diff(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("diff", args)
    n = args[1] if len(args) > 1 else kwargs.pop("n", 1)
    axis = kwargs.pop("axis", -1)
    if kwargs or len(args) > 2:
        raise TypeError("graphed np.diff takes (array, n, axis=)")
    normalized = a._norm_axis(int(axis))  # type: ignore[call-overload]
    params: dict[str, ParamValue] = {"axis": normalized if normalized is not None else 0}
    if int(n) != 1:  # type: ignore[call-overload]
        params["n"] = int(n)  # type: ignore[call-overload]
    return a.session.record_op("diff", [a], params)


def _fn_isin(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    if kwargs:
        raise TypeError("graphed np.isin takes (element array, test array)")
    a, b = _two_arrays("isin", args)
    return a.session.record_op("isin", [a, b])


def _fn_searchsorted(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    side = kwargs.pop("side", "left")
    if kwargs:
        raise TypeError("graphed np.searchsorted takes (sorted array, values array, side=)")
    a, v = _two_arrays("searchsorted", args)
    params: dict[str, ParamValue] = {"side": str(side)} if side != "left" else {}
    return a.session.record_op("searchsorted", [a, v], params)


def _fn_unique(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("unique", args)
    if kwargs or len(args) > 1:
        raise TypeError("graphed np.unique takes (array,)")
    return a.session.record_op("unique", [a], reduction=True)


def _fn_bincount(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("bincount", args)
    if kwargs or len(args) > 1:
        raise TypeError("graphed np.bincount takes (array,)")
    return a.session.record_op("bincount", [a], reduction=True)


def _hist_bins_range(kind: str, bins: object, rng: object) -> tuple[int, tuple[float, float]]:
    if isinstance(bins, bool) or not isinstance(bins, int) or rng is None:
        raise TypeError(
            f"graphed np.{kind} needs an int `bins` and an explicit `range` — the bin edges must "
            "be fully determined at record time (no data peeking in a deferred graph)"
        )
    if not isinstance(rng, list | tuple) or len(rng) != 2:
        raise TypeError(f"graphed np.{kind}: range must be a (lo, hi) pair")
    return bins, (float(rng[0]), float(rng[1]))


def _fn_histogram(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("histogram", args)
    bins = args[1] if len(args) > 1 else kwargs.pop("bins", 10)
    rng = kwargs.pop("range", None)
    if kwargs or len(args) > 2:
        raise TypeError("graphed np.histogram takes (array, bins, range=)")
    nbins, (lo, hi) = _hist_bins_range("histogram", bins, rng)
    counts = a.session.record_op("histogram", [a], {"bins": nbins, "lo": lo, "hi": hi}, reduction=True)
    return counts, np.histogram_bin_edges(np.empty(0), bins=nbins, range=(lo, hi))


def _fn_histogram2d(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    bins = kwargs.pop("bins", 10)
    rng = kwargs.pop("range", None)
    if kwargs or len(args) != 2 or not all(isinstance(a, NumpyArray) for a in args):
        raise TypeError("graphed np.histogram2d takes (x array, y array, bins=, range=)")
    if not isinstance(rng, list | tuple) or len(rng) != 2:
        raise TypeError("graphed np.histogram2d needs range=[(xlo,xhi),(ylo,yhi)]")
    x, y = _two_arrays("histogram2d", args)
    nbins, (xlo, xhi) = _hist_bins_range("histogram2d", bins, rng[0])
    _, (ylo, yhi) = _hist_bins_range("histogram2d", bins, rng[1])
    counts = x.session.record_op(
        "histogram2d",
        [x, y],
        {"bins": nbins, "xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi},
        reduction=True,
    )
    xedges = np.histogram_bin_edges(np.empty(0), bins=nbins, range=(xlo, xhi))
    yedges = np.histogram_bin_edges(np.empty(0), bins=nbins, range=(ylo, yhi))
    return counts, xedges, yedges


def _fn_histogramdd(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    a = _one_array("histogramdd", args)
    bins = kwargs.pop("bins", 10)
    rng = kwargs.pop("range", None)
    if kwargs or len(args) > 1:
        raise TypeError("graphed np.histogramdd takes (sample array, bins=, range=)")
    if isinstance(bins, bool) or not isinstance(bins, int) or rng is None:
        raise TypeError("graphed np.histogramdd needs an int `bins` and an explicit per-dim `range`")
    if not isinstance(rng, list | tuple):
        raise TypeError("graphed np.histogramdd needs a per-dim range sequence")
    pairs = [(float(lo), float(hi)) for lo, hi in rng]
    params: dict[str, ParamValue] = {
        "bins": bins,
        "los": ",".join(repr(lo) for lo, _ in pairs),
        "his": ",".join(repr(hi) for _, hi in pairs),
    }
    counts = a.session.record_op("histogramdd", [a], params, reduction=True)
    edges = [np.histogram_bin_edges(np.empty(0), bins=bins, range=p) for p in pairs]
    return counts, edges


_NUMPY_FUNCTIONS.update(
    {
        "reshape": _fn_reshape,
        "ravel": _fn_ravel,
        "squeeze": _fn_squeeze,
        "transpose": _fn_transpose,
        "swapaxes": _fn_swapaxes,
        "expand_dims": _fn_expand_dims,
        "clip": _fn_clip,
        "round": _fn_round,
        "around": _fn_round,
        "take": _fn_take,
        "astype": _fn_astype,
        "where": _fn_where,
        "concatenate": _fn_concatenate,
        "hstack": _fn_hstack,
        "stack": _fn_stack_refused,
        "vstack": _fn_stack_refused,
        "diff": _fn_diff,
        "isin": _fn_isin,
        "searchsorted": _fn_searchsorted,
        "unique": _fn_unique,
        "bincount": _fn_bincount,
        "histogram": _fn_histogram,
        "histogram2d": _fn_histogram2d,
        "histogramdd": _fn_histogramdd,
    }
)
