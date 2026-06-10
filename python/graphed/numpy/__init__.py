"""graphed-numpy: a numpy backend proving the graphed backend seam (plans M2 + M5 + M11).

Operates on numpy arrays whose axis 0 is the partitioned (unknown-length) axis: the full
elementwise/ufunc tier, boolean filtering, sum reduction, and arbitrary opaque Python callables via
`map`. M5 adds **record sources** (named columns) + field access and a real **necessary-buffer
(column) projection** that tracks which fields each computation touches. M11 (dask.array parity P0)
adds a real ``(shape, dtype)`` form inferred at record time by evaluating each op on zero-length
**meta arrays** — dtype promotion, broadcasting, and type errors are numpy's own (plan §A.2: reuse
the host library's inference) — and the ``NumpyArray`` proxy (``array.py``): the shared
``graphed.Array`` is backend-idiom-neutral, so the numpy calling idiom (metadata properties,
``__array_function__``) is completed HERE and handed to the Session via ``array_type``.
`external_payload` flags wrapped opaque callables as a preservation risk (plan A.3.1).
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from graphed import Array, Session
from graphed_core import PayloadDescriptor

from .array import NumpyArray
from .forms import NumpyForm, form_from_meta, is_numeric, meta, unit_meta
from .projection import project

# canonical op name -> numpy callable. These are the evaluation AND the record-time form-inference
# tables: op_form applies the same callable to zero-length meta arrays, so the two cannot drift.
_UNARY: dict[str, Any] = {
    "abs": np.absolute,
    "fabs": np.fabs,
    "neg": np.negative,
    "pos": np.positive,
    "sign": np.sign,
    "signbit": np.signbit,
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc,
    "rint": np.rint,
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "square": np.square,
    "reciprocal": np.reciprocal,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    "deg2rad": np.deg2rad,
    "rad2deg": np.rad2deg,
    "isnan": np.isnan,
    "isinf": np.isinf,
    "isfinite": np.isfinite,
    "spacing": np.spacing,
    "conj": np.conjugate,
    "invert": np.invert,
    "logical_not": np.logical_not,
}

_BINARY: dict[str, Any] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.true_divide,
    "floordiv": np.floor_divide,
    "mod": np.remainder,
    "power": np.power,
    "float_power": np.float_power,
    "fmod": np.fmod,
    "hypot": np.hypot,
    "arctan2": np.arctan2,
    "copysign": np.copysign,
    "nextafter": np.nextafter,
    "ldexp": np.ldexp,
    "fmax": np.fmax,
    "fmin": np.fmin,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "logaddexp": np.logaddexp,
    "logaddexp2": np.logaddexp2,
    "heaviside": np.heaviside,
    "gcd": np.gcd,
    "lcm": np.lcm,
    "gt": np.greater,
    "lt": np.less,
    "ge": np.greater_equal,
    "le": np.less_equal,
    "eq": np.equal,
    "ne": np.not_equal,
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
    "lshift": np.left_shift,
    "rshift": np.right_shift,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_xor": np.logical_xor,
}

# reductions (boundary when over the partitioned axis) and scans (always partition-local). The
# same callables serve record-time inference (on LENGTH-ONE unit metas — argmin/mean reject empty
# input) and evaluation, so the two cannot drift (M12).
_REDUCERS: dict[str, Any] = {
    "sum": np.sum,
    "prod": np.prod,
    "mean": np.mean,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "any": np.any,
    "all": np.all,
    "argmin": np.argmin,
    "argmax": np.argmax,
    "nansum": np.nansum,
    "nanprod": np.nanprod,
    "nanmean": np.nanmean,
    "nanstd": np.nanstd,
    "nanvar": np.nanvar,
    "nanmin": np.nanmin,
    "nanmax": np.nanmax,
    "nanargmin": np.nanargmin,
    "nanargmax": np.nanargmax,
}

_SCANS: dict[str, Any] = {
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "nancumsum": np.nancumsum,
    "nancumprod": np.nancumprod,
}

# *_like creators are OPS (the operand's partitioned length is unknown at record time); empty_like
# is recorded as zeros_like — uninitialized memory would break the byte-identity determinism gate.
_LIKE: dict[str, Any] = {
    "zeros_like": np.zeros_like,
    "ones_like": np.ones_like,
    "full_like": np.full_like,
}

# M13 manipulation/combination ops (dask.array parity P2). Partition-local ones are fusible;
# slice/index/take(axis 0)/concatenate(axis 0) and the whole-axis analytics are boundaries —
# the RECORDING side (graphed Array / NumpyArray) decides that; here is evaluation + inference.
_MANIP_OPS = frozenset(
    {
        "slice",
        "index",
        "getitem",
        "subscript",
        "reshape",
        "ravel",
        "squeeze",
        "expand_dims",
        "swapaxes",
        "transpose",
        "astype",
        "clip",
        "round",
        "take",
        "where",
        "concatenate",
        "diff",
        "isin",
        "searchsorted",
        "unique",
        "bincount",
        "histogram",
        "histogram2d",
        "histogramdd",
    }
)

# analytics whose meta result is already concrete (the partitioned axis is consumed)
_CONCRETE_AGGS = frozenset({"histogram", "histogram2d", "histogramdd"})


def _decode_dims(spec: object) -> tuple[int, ...]:
    return tuple(int(p) for p in str(spec).split(",")) if str(spec) else ()


def _decode_subscript(spec: object) -> tuple[Any, ...]:
    out: list[Any] = []
    for part in str(spec).split(","):
        if ":" in part:
            bits = part.split(":")
            out.append(slice(*(int(b) if b else None for b in bits)))
        else:
            out.append(int(part))
    return tuple(out)


def _manip_eval(op: str, xs: Sequence[Any], params: Mapping[str, object]) -> Any:
    """Evaluate one M13 op — shared by record-time inference (on metas) and eval_stage."""
    if op == "slice":
        key = slice(*(params.get(k) for k in ("start", "stop", "step")))
        return np.asarray(xs[0])[key]
    if op == "index":
        return np.asarray(xs[0])[int(params["i"])]  # type: ignore[call-overload]
    if op == "getitem":
        return np.asarray(xs[0])[np.asarray(xs[1])]
    if op == "subscript":
        return np.asarray(xs[0])[_decode_subscript(params["spec"])]
    if op == "reshape":
        return np.reshape(xs[0], _decode_dims(params["shape"]))
    if op == "ravel":
        return np.ravel(xs[0])
    if op == "squeeze":
        return np.squeeze(xs[0], axis=int(params["axis"]))  # type: ignore[call-overload]
    if op == "expand_dims":
        return np.expand_dims(xs[0], int(params["axis"]))  # type: ignore[call-overload]
    if op == "swapaxes":
        return np.swapaxes(xs[0], int(params["a1"]), int(params["a2"]))  # type: ignore[call-overload]
    if op == "transpose":
        return np.transpose(xs[0], _decode_dims(params["axes"]) if "axes" in params else None)
    if op == "astype":
        return np.asarray(xs[0]).astype(np.dtype(str(params["dtype"])))
    if op == "clip":
        return np.clip(xs[0], params.get("lo"), params.get("hi"))
    if op == "round":
        return np.round(xs[0], int(params.get("decimals", 0)))  # type: ignore[call-overload]
    if op == "take":
        axis = params.get("axis")
        return np.take(xs[0], np.asarray(xs[1]).astype(np.intp), axis=None if axis is None else int(axis))  # type: ignore[call-overload]
    if op == "where":
        rest = list(xs[1:])
        xval = params["x_scalar"] if "x_scalar" in params else rest.pop(0)
        yval = params["y_scalar"] if "y_scalar" in params else rest.pop(0)
        return np.where(np.asarray(xs[0]), xval, yval)
    if op == "concatenate":
        return np.concatenate([np.asarray(x) for x in xs], axis=int(params.get("axis", 0)))  # type: ignore[call-overload]
    if op == "diff":
        return np.diff(xs[0], n=int(params.get("n", 1)), axis=int(params.get("axis", -1)))  # type: ignore[call-overload]
    if op == "isin":
        return np.isin(xs[0], xs[1])
    if op == "searchsorted":
        return np.searchsorted(xs[0], xs[1], side=str(params.get("side", "left")))  # type: ignore[call-overload]
    if op == "unique":
        return np.unique(xs[0])
    if op == "bincount":
        return np.bincount(np.asarray(xs[0]))
    if op == "histogram":
        rng = (float(params["lo"]), float(params["hi"]))  # type: ignore[arg-type]
        return np.histogram(xs[0], bins=int(params["bins"]), range=rng)[0]  # type: ignore[call-overload]
    if op == "histogram2d":
        rng2 = [
            (float(params["xlo"]), float(params["xhi"])),  # type: ignore[arg-type]
            (float(params["ylo"]), float(params["yhi"])),  # type: ignore[arg-type]
        ]
        return np.histogram2d(xs[0], xs[1], bins=int(params["bins"]), range=rng2)[0]  # type: ignore[call-overload]
    if op == "histogramdd":
        lows = [float(v) for v in str(params["los"]).split(",")]
        highs = [float(v) for v in str(params["his"]).split(",")]
        rngs = list(zip(lows, highs, strict=True))
        return np.histogramdd(xs[0], bins=int(params["bins"]), range=rngs)[0]  # type: ignore[call-overload]
    raise TypeError(f"unsupported op {op!r}")


def _check_manip_geometry(op: str, forms: Sequence[NumpyForm], params: Mapping[str, object]) -> None:
    """The axis-0-partitioned MVP's geometry rules, enforced at record time (Phase 2 lifts them)."""
    if op == "reshape":
        dims = _decode_dims(params["shape"])
        if not dims or dims[0] != -1 or any(d <= 0 for d in dims[1:]):
            raise TypeError(
                "reshape in the axis-0-partitioned MVP needs shape (-1, concrete...): the "
                "partitioned axis must stay leading and its length is unknown at record time"
            )
    elif op == "squeeze" and int(params["axis"]) == 0:  # type: ignore[call-overload]
        raise TypeError("cannot squeeze the partitioned axis 0")
    elif op == "expand_dims" and int(params["axis"]) == 0:  # type: ignore[call-overload]
        raise TypeError("cannot displace the partitioned axis 0 (expand an inner axis instead)")
    elif op == "swapaxes" and 0 in (int(params["a1"]), int(params["a2"])):  # type: ignore[call-overload]
        raise TypeError("cannot move the partitioned axis 0 (swap inner axes instead)")
    elif op == "transpose":
        if "axes" in params:
            if _decode_dims(params["axes"])[0] != 0:
                raise TypeError("transpose must keep the partitioned axis 0 in place")
        elif forms[0].ndim > 1:
            raise TypeError(
                "transpose without axes reverses them, displacing the partitioned axis 0; "
                "pass explicit axes keeping axis 0 first"
            )


def _reduce_kwargs(params: Mapping[str, object]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "axis" in params:
        kwargs["axis"] = params["axis"]
    if params.get("keepdims"):
        kwargs["keepdims"] = True
    if "ddof" in params:
        kwargs["ddof"] = params["ddof"]
    return kwargs


class NumpyBackend:
    """A `graphed.Backend` over numpy arrays: M2 seam + M5 records + the M11 elementwise tier."""

    def array_type(self) -> type[Array]:
        """The numpy-idiomatic proxy (M11 factorization): Sessions return ``NumpyArray``."""
        return NumpyArray

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> NumpyForm:
        forms = [f for f in inputs if isinstance(f, NumpyForm)]
        if op == "field":
            (rec,) = forms
            if rec.fields is None:
                raise TypeError(f"field access requires a record source, got {rec.describe()}")
            fd = dict(rec.fields)
            name = str(params["field"])
            if name not in fd:
                raise TypeError(f"record has no field {name!r}; fields are {sorted(fd)}")
            return NumpyForm(np.dtype(fd[name]))
        if op == "filter":
            data, mask = forms
            if mask.dtype != np.bool_:
                raise TypeError(f"filter mask must be boolean, got {mask.describe()}")
            return NumpyForm(data.dtype, shape=data.shape)
        if op == "map":
            return NumpyForm(np.dtype(object))  # opaque callable: result form unknown
        if op in _REDUCERS or op in _SCANS:
            (a,) = forms
            if not (is_numeric(a) or (a.fields is None and a.dtype == np.bool_)):
                raise TypeError(f"{op} requires a numeric array, got {a.describe()}")
            result = (_REDUCERS | _SCANS)[op](unit_meta(a), **_reduce_kwargs(params))
            axis = params.get("axis")
            # scans and inner-axis reductions preserve the partitioned axis; axis None/0 consume it
            leading_none = op in _SCANS or (axis is not None and axis != 0)
            return form_from_meta(result, leading_none)
        if op in _MANIP_OPS:
            _check_manip_geometry(op, forms, params)
            if op == "index":  # length-one stand-in; row 0 stands for any row
                result = _manip_eval(op, [unit_meta(forms[0])], {"i": 0})
                return form_from_meta(result, False)
            if op == "take":  # the gathered extent equals the (partitioned) index length
                _manip_eval(op, [meta(f) for f in forms], params)  # validate evaluability
                data, _ = forms
                axis_p = params.get("axis")
                k = 0 if axis_p is None else int(axis_p)  # type: ignore[call-overload]
                if axis_p is None:
                    return NumpyForm(data.dtype, shape=(None,))
                return NumpyForm(data.dtype, shape=(*data.shape[:k], None, *data.shape[k + 1 :]))
            result = _manip_eval(op, [meta(f) for f in forms], params)
            leading_none = op not in _CONCRETE_AGGS
            return form_from_meta(result, leading_none)
        result = self._apply(op, [meta(f) for f in forms], params)
        leading_none = any(f.shape and f.shape[0] is None for f in forms)
        return form_from_meta(result, leading_none)

    def _apply(self, op: str, operands: Sequence[Any], params: Mapping[str, object]) -> Any:
        """Evaluate one elementwise op — shared by record-time inference (on metas) and eval."""
        if op in _LIKE and "scalar" not in params:
            (x,) = operands
            if op == "full_like":
                return np.full_like(x, params["fill"])
            return _LIKE[op](x)
        if op in _UNARY and "scalar" not in params:
            (x,) = operands
            return _UNARY[op](x)
        if op in _BINARY:
            if "scalar" in params:
                (x,) = operands
                s = params["scalar"]
                return _BINARY[op](s, x) if params.get("side") == "l" else _BINARY[op](x, s)
            a, b = operands
            return _BINARY[op](a, b)
        raise TypeError(f"unsupported op {op!r}")

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        if op == "field":
            return np.asarray(inputs[0][str(params["field"])])  # type: ignore[index]
        if op == "filter":
            return np.asarray(inputs[0])[np.asarray(inputs[1])]
        if op in _REDUCERS or op in _SCANS:
            return (_REDUCERS | _SCANS)[op](np.asarray(inputs[0]), **_reduce_kwargs(params))
        if op in _MANIP_OPS:
            return _manip_eval(op, [np.asarray(x) for x in inputs], params)
        return self._apply(op, [np.asarray(x) for x in inputs], params)

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source", "map", *_REDUCERS})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        # Vestigial M2 per-op stub; the real projection is the module-level `project` (M5).
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        if op != "map":
            return None
        fn_name = str(params.get("fn", "lambda"))
        return PayloadDescriptor(
            kind="opaque_callable",
            content_hash=f"unhashed-opaque:{fn_name}",
            framework="python",
            version=platform.python_version(),
            io_schema="opaque->opaque",
            preprocessing_ref=None,
        )


def from_array(session: Session, name: str, values: object, *, chunks: int | None = None) -> Array:
    """Create a source Array from a numpy array (or anything array-like); axis 0 is partitioned.

    ``chunks`` is axis-0 partitioning METADATA for executors (dask parity P1.5): it enters the
    source's recorded params (and therefore the IR identity) but does not change evaluation.
    """
    arr = np.asarray(values)
    form = NumpyForm(arr.dtype, shape=(None, *arr.shape[1:]))
    if chunks is None:
        return session.source(name, form=form, data=arr)
    return session.source(name, form=form, data=arr, chunks=int(chunks))


def from_record(session: Session, name: str, **columns: object) -> Array:
    """Create a record source (named columns) — the buffer-projection target."""
    cols = {k: np.asarray(v) for k, v in columns.items()}
    fields = tuple((k, v.dtype.str) for k, v in cols.items())
    return session.source(name, form=NumpyForm(np.dtype(object), kind="record", fields=fields), data=cols)


# M12 (dask.array parity P1): creation routines, deterministic random sources, and the
# tree-reducible reduction monoids that drop into the M7 process/combine/empty model.
from .creation import (  # noqa: E402
    arange,
    empty,
    empty_like,
    full,
    full_like,
    linspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .random import GraphedRNG, default_rng  # noqa: E402
from .reductions import Monoid, monoid  # noqa: E402

__all__ = [
    "GraphedRNG",
    "Monoid",
    "NumpyArray",
    "NumpyBackend",
    "NumpyForm",
    "arange",
    "default_rng",
    "empty",
    "empty_like",
    "from_array",
    "from_record",
    "full",
    "full_like",
    "linspace",
    "monoid",
    "ones",
    "ones_like",
    "project",
    "zeros",
    "zeros_like",
]
__version__ = "0.0.1"
