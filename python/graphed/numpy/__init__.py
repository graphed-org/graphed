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
