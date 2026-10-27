"""The awkward op dispatch shared by form inference (typetracer) and evaluation (real arrays).

`apply(op, operands, params)` runs the actual awkward/numpy operation. The *same* function drives
`op_form` (operands are metadata-only typetracer arrays — no data is read) and `eval_stage`
(operands are real arrays). This guarantees forms and results come from one source of truth.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np

from graphed.array import check_leading_ellipsis, decode_subscript
from graphed.errors import GraphedError

from . import join

# Elementwise ops: canonical name -> callable. Unary take 1 operand, binary take 2.
# M16 (parity plan P0): the FULL M11 canonical ufunc tier — awkward arrays take numpy ufuncs
# directly, and the typetracer infers forms for every one of them.
_UNARY: dict[str, Any] = {
    "abs": np.abs,
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
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "power": lambda a, b: a**b,
    "mod": lambda a, b: a % b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "hypot": np.hypot,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "floordiv": np.floor_divide,
    "fmod": np.fmod,
    "float_power": np.float_power,
    "arctan2": np.arctan2,
    "copysign": np.copysign,
    "nextafter": np.nextafter,
    "ldexp": np.ldexp,
    "fmax": np.fmax,
    "fmin": np.fmin,
    "logaddexp": np.logaddexp,
    "logaddexp2": np.logaddexp2,
    "heaviside": np.heaviside,
    "gcd": np.gcd,
    "lcm": np.lcm,
    "xor": np.bitwise_xor,
    "lshift": np.left_shift,
    "rshift": np.right_shift,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_xor": np.logical_xor,
}

# Reductions over one array, all honoring the M12/M16 structural rule at RECORD time (the
# recording side decides boundary-vs-fusible; this table only evaluates).
_AK_REDUCERS: dict[str, Any] = {
    "ak.sum": ak.sum,
    "ak.any": ak.any,
    "ak.all": ak.all,
    "ak.count": ak.count,
    "ak.count_nonzero": ak.count_nonzero,
    "ak.min": ak.min,
    "ak.max": ak.max,
    "ak.prod": ak.prod,
    "ak.mean": ak.mean,
    "ak.ptp": ak.ptp,
}

_AK_TWO_INPUT: dict[str, Any] = {
    "ak.corr": ak.corr,
    "ak.covar": ak.covar,
    "ak.linear_fit": ak.linear_fit,
}


def scalar_param(params: Mapping[str, Any], key: str = "scalar", dtype_key: str = "dtype") -> Any:
    """Return the scalar operand a binary op recorded.

    Back in its own dtype when it had one (M59). The frontend cannot build a numpy scalar, so it
    records the dtype name beside the value, and a value wider than the store's i64 param as text.
    """
    value = params[key]
    if dtype_key not in params:
        return value
    # the dtype parses the wide int's text itself
    return np.dtype(str(params[dtype_key])).type(value)


def _scalar_operands(operands: Sequence[Any], params: Mapping[str, Any]) -> list[Any]:
    """Reconstruct positional operands when one was a scalar (encoded in params)."""
    if "scalar" in params:
        s = scalar_param(params)
        return [s, operands[0]] if params.get("side") == "l" else [operands[0], s]
    return list(operands)


def _leaves(result: Any) -> list[Any]:
    """Return a method result's arrays, depth-first.

    That is the order its `index` param numbers them in.
    """
    if isinstance(result, tuple):
        return [leaf for item in result for leaf in _leaves(item)]
    return [result]


def _fields(params: Mapping[str, Any]) -> list[str]:
    raw = params.get("fields", "")
    return [f for f in str(raw).split(",") if f]


def _structure(x: Any) -> str:
    """Return a value's type BELOW its outer length.

    The part a typetracer form and a real partition render identically, so one blame message reads
    the same on either path.
    """
    return str(ak.Array(x).type).split(" * ", 1)[-1]


def _broadcast_blame(operands: Sequence[Any], exc: ValueError) -> Exception:
    """§6.1d: translate awkward's two-anonymous-layouts complaint into a `graphed` error.

    The error names the offending FACTOR — the operand `broadcast_like` broadcasts to the value's
    structure.

    One site for both paths: a regular-vs-regular mismatch is decidable from the typetracer forms
    and lands at record time (where `Session.record_op` re-raises this at the user's line), while a
    jagged pair whose counts differ only in the data lands at execution time.
    """
    value, factors = _structure(operands[0]), [_structure(f) for f in operands[1:]]
    # Two LEAF structures can only disagree on outer length, which is the already-flattened-value
    # case §6.1d names: a per-event factor against a per-object value someone flattened first.
    unflatten = (
        " — a per-event factor against a per-object value needs the value passed UNFLATTENED"
        if " * " not in value and all(" * " not in f for f in factors)
        else ""
    )
    return GraphedError(
        f"cannot broadcast the factor ({', '.join(factors)}) to the value's structure "
        f"({value}){unflatten}: {exc}"
    )


def _decode_call(text: str, operands: Sequence[Any]) -> Any:
    """Rebuild a `method` op's argument structure.

    `{"$": i}` becomes operand `i` (M54) and `{"$dtype": name, "v": item}` the numpy scalar it was
    recorded from.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"$"}:
                return operands[int(value["$"])]
            if set(value) == {"$dtype", "v"}:
                return np.dtype(value["$dtype"]).type(value["v"])
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(json.loads(text))


def apply(
    op: str,
    operands: Sequence[Any],
    params: Mapping[str, Any],
    behavior: Mapping[str, Any] | None = None,
) -> Any:
    if behavior:
        # M54: the backend's behavior dict rides on every operand (with its own attrs), so a
        # behavior registered on the backend alone (not in the global ak.behavior) resolves its
        # properties, methods and operator overloads exactly like a globally registered one
        operands = [
            ak.Array(x.layout, behavior=behavior, attrs=x.attrs) if isinstance(x, ak.Array) else x
            for x in operands
        ]
    if op in _UNARY:
        return _UNARY[op](operands[0])
    if op in _BINARY:
        a, b = _scalar_operands(operands, params)
        return _BINARY[op](a, b)
    if op == "ak.with_name":
        named = ak.with_name(operands[0], str(params["name"]))
        if behavior:
            return ak.Array(named.layout, behavior=behavior, attrs=named.attrs)
        return named
    handler = _OPS.get(op)
    if handler is None:
        raise TypeError(f"unsupported awkward op {op!r}")
    return handler(op, operands, params)


#: An op's evaluation: ``(op, operands, params) -> value``, on typetracers and real arrays alike.
_Handler = Callable[[str, Sequence[Any], Mapping[str, Any]], Any]


def _field(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    x = xs[0]
    name = str(p["field"])
    if name in getattr(x, "fields", []):
        return x[name]
    # M18: not a record field -> a behavior property (vector's .pt/.mass, the coffea pattern);
    # works on the typetracer and real arrays alike when the array carries the behavior. An
    # unknown attribute raises here -> a record-time GraphedTypeError.
    return getattr(x, name)


def _method(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    # M54: a behavior METHOD called with arguments. Array-valued arguments arrived as operands
    # and stand in the JSON as {"$": index}; constants are plain JSON. Typetracer and real
    # arrays share this branch, so form inference, projection replay and evaluation agree.
    target = getattr(xs[0], str(p["method"]))
    result = target(*_decode_call(str(p["args"]), xs), **_decode_call(str(p["kwargs"]), xs))
    # M59: a tuple result may be NESTED; `index` numbers its leaves depth-first
    return _leaves(result)[int(p["index"])] if "index" in p else result


def _subscript(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    # M59: the shared inner-axis tuple key (axis 0 left whole)
    key = decode_subscript(p["spec"])
    # awkward's depth here is the MINIMUM list depth: `ndim` reads a record as depth 1 even
    # when every field is a list, and `[..., 0]` does reach into those fields
    check_leading_ellipsis(key, xs[0].layout.minmax_depth[0])
    return xs[0][key]


def _reducer(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    if (
        op in ("ak.min", "ak.max", "ak.ptp")
        and _axis(p) is None
        and not any(k in p for k in ("mask_identity", "initial", "keepdims"))
    ):
        return _global_extremum(op, xs[0])
    return _AK_REDUCERS[op](xs[0], axis=_axis(p), **_reduce_kwargs(op, p, xs))


def _std_var(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    if (
        _axis(p) is None
        and not any(k in p for k in ("mask_identity", "keepdims"))
        and ak.backend(xs[0]) == "typetracer"
    ):
        return _global_moment(op, p, xs)
    fn = ak.std if op == "ak.std" else ak.var
    return fn(xs[0], axis=_axis(p), ddof=int(p.get("ddof", 0)), **_reduce_kwargs(op, p, xs))


def _combinations(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    fn = ak.combinations if op == "ak.combinations" else ak.argcombinations
    return fn(
        xs[0],
        int(p["n"]),
        fields=_fields(p) or None,
        replacement=bool(p.get("replacement", False)),
        axis=int(p.get("axis", 1)),
        **_naming_kwargs(p),
    )


def _cartesian(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    fn = ak.cartesian if op == "ak.cartesian" else ak.argcartesian
    keys = _fields(p)
    nested = _nested(p)
    return fn(
        dict(zip(keys, xs, strict=True)) if keys else list(xs),
        nested=[keys[i] for i in nested] if keys and isinstance(nested, list) else nested,
        axis=int(p.get("axis", 1)),
        **_naming_kwargs(p),
    )


def _zip(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    zipped = tuple(xs) if p.get("tuple") else dict(zip(_fields(p), xs, strict=True))
    kw: dict[str, Any] = dict(_naming_kwargs(p))
    if "depth_limit" in p:
        kw["depth_limit"] = int(p["depth_limit"])
    return ak.zip(zipped, **kw)


def _arg_extremum(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    fn = ak.argmin if op == "ak.argmin" else ak.argmax
    return fn(
        xs[0],
        axis=_axis(p),
        keepdims=bool(p.get("keepdims", False)),
        mask_identity=bool(p.get("mask_identity", True)),
    )


def _sorting(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    fn = ak.sort if op == "ak.sort" else ak.argsort
    return fn(
        xs[0],
        axis=int(p.get("axis", -1)),
        ascending=bool(p.get("ascending", True)),
        stable=bool(p.get("stable", True)),
    )


def _drop_none(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    ax = p.get("axis")
    return ak.drop_none(xs[0], axis=None if ax is None else int(ax))


def _nan_to_num(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    return ak.nan_to_num(
        xs[0],
        nan=float(p.get("nan", 0.0)),
        posinf=float(p["posinf"]) if "posinf" in p else None,
        neginf=float(p["neginf"]) if "neginf" in p else None,
    )


def _isclose(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    return ak.isclose(
        xs[0],
        xs[1],
        rtol=float(p.get("rtol", 1e-05)),
        atol=float(p.get("atol", 1e-08)),
        equal_nan=bool(p.get("equal_nan", False)),
    )


def _broadcast_arrays(op: str, xs: Sequence[Any], p: Mapping[str, Any]) -> Any:
    dl = int(p["depth_limit"]) if "depth_limit" in p else None
    try:
        return ak.broadcast_arrays(*xs, depth_limit=dl)[int(p["index"])]
    except ValueError as exc:  # §6.1d: awkward names two anonymous layouts; blame the factor
        raise _broadcast_blame(xs, exc) from exc


def _like_dtype(p: Mapping[str, Any]) -> Any:
    return _dtype(p) if "dtype" in p else None


_OPS: dict[str, _Handler] = {
    # M40 §2.1: add the big-endian-packed u64 __joinkey__ column
    "pack_key": lambda op, xs, p: join.pack_key(xs[0], join.on_from_params(p)),
    # a pure data-movement boundary is identity on values in-process (§3.3a)
    "exchange": lambda op, xs, p: xs[0],
    "field": _field,
    "method": _method,
    "getitem": lambda op, xs, p: xs[0][xs[1]],
    "filter": lambda op, xs, p: xs[0][xs[1]],
    # the M13 common axis-0 slice (start/stop/step present-only)
    "slice": lambda op, xs, p: xs[0][slice(p.get("start"), p.get("stop"), p.get("step"))],
    "subscript": _subscript,
    "index": lambda op, xs, p: xs[0][int(p["i"])],  # the M13 common integer index
    "fields": lambda op, xs, p: xs[0][[f for f in str(p["fields"]).split(",") if f]],
    "ak.num": lambda op, xs, p: ak.num(xs[0], axis=int(p.get("axis", 1))),
    **dict.fromkeys(_AK_REDUCERS, _reducer),
    "ak.std": _std_var,
    "ak.var": _std_var,
    "ak.moment": lambda op, xs, p: ak.moment(
        xs[0], int(p["n"]), axis=_axis(p), **_reduce_kwargs(op, p, xs)
    ),
    "ak.softmax": lambda op, xs, p: ak.softmax(xs[0], axis=int(p.get("axis", 1))),
    **{
        name: lambda op, xs, p: _AK_TWO_INPUT[op](xs[0], xs[1], axis=_axis(p))
        for name in _AK_TWO_INPUT
    },
    "ak.combinations": _combinations,
    "ak.argcombinations": _combinations,
    "ak.cartesian": _cartesian,
    "ak.argcartesian": _cartesian,
    "ak.zip": _zip,
    "ak.with_field": lambda op, xs, p: ak.with_field(
        xs[0], scalar_param(p) if "scalar" in p else xs[1], where=p["field"]
    ),
    "ak.firsts": lambda op, xs, p: ak.firsts(xs[0], axis=int(p.get("axis", 1))),
    "ak.argmin": _arg_extremum,
    "ak.argmax": _arg_extremum,
    "ak.argsort": _sorting,
    "ak.sort": _sorting,
    "ak.local_index": lambda op, xs, p: ak.local_index(xs[0], axis=int(p.get("axis", -1))),
    "ak.concatenate": lambda op, xs, p: ak.concatenate(list(xs), axis=int(p.get("axis", 0))),
    "ak.flatten": lambda op, xs, p: ak.flatten(xs[0], axis=_axis(p, default=1)),
    "ak.fill_none": lambda op, xs, p: ak.fill_none(
        xs[0], scalar_param(p, "value", "value_dtype"), axis=int(p.get("axis", -1))
    ),
    "ak.drop_none": _drop_none,
    "ak.where": lambda op, xs, p: ak.where(*_three_operands(xs, p)),
    "ak.zeros_like": lambda op, xs, p: ak.zeros_like(xs[0], dtype=_like_dtype(p)),
    "ak.ones_like": lambda op, xs, p: ak.ones_like(xs[0], dtype=_like_dtype(p)),
    "ak.values_astype": lambda op, xs, p: ak.values_astype(xs[0], _dtype(p)),
    "ak.with_parameter": lambda op, xs, p: ak.with_parameter(xs[0], str(p["key"]), p["value"]),
    "ak.without_parameters": lambda op, xs, p: ak.without_parameters(xs[0]),
    "ak.ravel": lambda op, xs, p: ak.ravel(xs[0]),
    "ak.run_lengths": lambda op, xs, p: ak.run_lengths(xs[0]),
    "ak.mask": lambda op, xs, p: ak.mask(xs[0], xs[1], valid_when=bool(p.get("valid_when", True))),
    "ak.is_none": lambda op, xs, p: ak.is_none(xs[0], axis=int(p.get("axis", 0))),
    "ak.singletons": lambda op, xs, p: ak.singletons(xs[0], axis=int(p.get("axis", 0))),
    "ak.pad_none": lambda op, xs, p: ak.pad_none(
        xs[0], int(p["target"]), axis=int(p.get("axis", 1)), clip=bool(p.get("clip", False))
    ),
    "ak.unflatten": lambda op, xs, p: ak.unflatten(
        xs[0], xs[1] if len(xs) > 1 else int(p["counts"]), axis=int(p.get("axis", 0))
    ),
    "ak.to_regular": lambda op, xs, p: ak.to_regular(xs[0], axis=int(p.get("axis", 1))),
    "ak.from_regular": lambda op, xs, p: ak.from_regular(xs[0], axis=int(p.get("axis", 1))),
    "ak.full_like": lambda op, xs, p: ak.full_like(
        xs[0],
        scalar_param(p, "value", "value_dtype"),
        dtype=np.dtype(str(p["dtype"])) if "dtype" in p else None,
    ),
    "ak.nan_to_num": _nan_to_num,
    "ak.isclose": _isclose,
    "ak.without_field": lambda op, xs, p: ak.without_field(xs[0], p["field"]),
    "ak.broadcast_arrays": _broadcast_arrays,
}


def _global_extremum(op: str, x: Any) -> Any:
    """axis=None min/max/ptp need typetracer-aware handling.

    The typetracer's option-scalar (MaybeNone) cannot be formed or subtracted, and upstream
    ak.ptp(axis=None) indexes its scalar result. On the TYPETRACER path use awkward's own
    mask_identity=False inference (a plain unknown scalar); the REAL path keeps awkward's default
    semantics untouched. ptp composes max - min on the typetracer (identical kernels — not
    hand-rolled inference).
    """
    tracing = ak.backend(x) == "typetracer"
    if op == "ak.ptp":
        if tracing:
            high = ak.max(x, axis=None, mask_identity=False)
            low = ak.min(x, axis=None, mask_identity=False)
            return high - low
        return ak.ptp(x, axis=None)
    fn = ak.min if op == "ak.min" else ak.max
    return fn(x, axis=None, mask_identity=False) if tracing else fn(x, axis=None)


def _global_moment(op: str, params: Mapping[str, Any], operands: Sequence[Any]) -> Any:
    """axis=None std/var need the same typetracer-aware handling as _global_extremum.

    awkward 2.13.0's ak_var._impl indexes its mean scalar (`xmean[(0,) * xmean.ndim]`), which
    to_layout refuses for the tracer's unknown scalar whatever mask_identity says. Compose the
    abstract result from typetracer-safe kernels with awkward's own dtype rules (float32 promotes
    to float64 on both paths, measured); the REAL path keeps ak.std/ak.var semantics untouched. A
    weighted moment needs no special arm: the weight cannot change a scalar reduction's abstract
    form, and the real path still applies it.
    """
    x = operands[0]
    m1 = ak.mean(x, axis=None, mask_identity=False)
    var = ak.mean(x * x, axis=None, mask_identity=False) - m1 * m1
    ddof = int(params.get("ddof", 0))
    if ddof:
        n = ak.count(x, axis=None, mask_identity=False)
        var = var * (n / (n - ddof))
    return var if op == "ak.var" else np.sqrt(var)


def _reduce_kwargs(op: str, params: Mapping[str, Any], operands: Sequence[Any]) -> dict[str, Any]:
    """ak-parity reducer kwargs, present-only in params (ak's own defaults otherwise).

    A weighted moment carries its weight as the SECOND OPERAND.
    """
    kw: dict[str, Any] = {}
    if params.get("keepdims"):
        kw["keepdims"] = True
    if "mask_identity" in params:
        kw["mask_identity"] = bool(params["mask_identity"])
    if "initial" in params:
        kw["initial"] = scalar_param(params, "initial", "initial_dtype")
    if params.get("weighted"):
        kw["weight"] = operands[1]
    return kw


def _naming_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """Inline with_name/parameters for the structure constructors (JSON-decoded)."""
    kw: dict[str, Any] = {}
    if "with_name" in params:
        kw["with_name"] = str(params["with_name"])
    if "parameters" in params:
        kw["parameters"] = json.loads(str(params["parameters"]))
    return kw


def _nested(params: Mapping[str, Any]) -> Any:
    if "nested_axes" in params:
        return [int(x) for x in str(params["nested_axes"]).split(",") if x]
    if "nested" in params:
        return bool(params["nested"])
    return None


def _axis(params: Mapping[str, Any], default: int | None = None) -> int | None:
    if "axis" in params:
        a = params["axis"]
        return None if a == "none" else int(a)
    return default


def _dtype(params: Mapping[str, Any]) -> Any:
    return np.dtype(str(params.get("dtype", "float64")))


def _three_operands(operands: Sequence[Any], params: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """Reconstruct ak.where's (cond, a, b) where some args may be scalars (encoded by index)."""
    out: list[Any] = []
    it = iter(operands)
    for i in range(3):
        key = f"const{i}"
        out.append(scalar_param(params, key, f"{key}_dtype") if key in params else next(it))
    return out[0], out[1], out[2]
