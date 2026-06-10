"""The awkward op dispatch shared by form inference (typetracer) and evaluation (real arrays).

`apply(op, operands, params)` runs the actual awkward/numpy operation. The *same* function drives
`op_form` (operands are metadata-only typetracer arrays — no data is read) and `eval_stage`
(operands are real arrays). This guarantees forms and results come from one source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np

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


def _scalar_operands(operands: Sequence[Any], params: Mapping[str, Any]) -> list[Any]:
    """Reconstruct positional operands when one was a scalar (encoded in params)."""
    if "scalar" in params:
        s = params["scalar"]
        return [s, operands[0]] if params.get("side") == "l" else [operands[0], s]
    return list(operands)


def _fields(params: Mapping[str, Any]) -> list[str]:
    raw = params.get("fields", "")
    return [f for f in str(raw).split(",") if f]


def apply(op: str, operands: Sequence[Any], params: Mapping[str, Any]) -> Any:
    if op in _UNARY:
        return _UNARY[op](operands[0])
    if op in _BINARY:
        a, b = _scalar_operands(operands, params)
        return _BINARY[op](a, b)
    if op == "field":
        return operands[0][params["field"]]
    if op in ("getitem", "filter"):
        return operands[0][operands[1]]
    if op == "ak.num":
        return ak.num(operands[0], axis=int(params.get("axis", 1)))
    if op in ("ak.min", "ak.max", "ak.ptp") and _axis(params) is None:
        return _global_extremum(op, operands[0])
    if op in _AK_REDUCERS:
        return _AK_REDUCERS[op](operands[0], axis=_axis(params))
    if op in ("ak.std", "ak.var"):
        fn = ak.std if op == "ak.std" else ak.var
        return fn(operands[0], axis=_axis(params), ddof=int(params.get("ddof", 0)))
    if op == "ak.moment":
        return ak.moment(operands[0], int(params["n"]), axis=_axis(params))
    if op == "ak.softmax":
        return ak.softmax(operands[0], axis=int(params.get("axis", 1)))
    if op in _AK_TWO_INPUT:
        return _AK_TWO_INPUT[op](operands[0], operands[1], axis=_axis(params))
    if op == "ak.combinations":
        return ak.combinations(operands[0], int(params["n"]), fields=_fields(params) or None)
    if op == "ak.cartesian":
        return ak.cartesian(list(operands), nested=bool(params.get("nested", False)))
    if op == "ak.zip":
        return ak.zip(dict(zip(_fields(params), operands, strict=True)))
    if op == "ak.with_field":
        return ak.with_field(operands[0], operands[1], where=params["field"])
    if op == "ak.firsts":
        return ak.firsts(operands[0], axis=int(params.get("axis", 1)))
    if op == "ak.argmin":
        return ak.argmin(
            operands[0], axis=int(params.get("axis", 1)), keepdims=bool(params.get("keepdims", False))
        )
    if op == "ak.argmax":
        return ak.argmax(
            operands[0], axis=int(params.get("axis", 1)), keepdims=bool(params.get("keepdims", False))
        )
    if op == "ak.argsort":
        return ak.argsort(
            operands[0], axis=int(params.get("axis", 1)), ascending=bool(params.get("ascending", True))
        )
    if op == "ak.local_index":
        return ak.local_index(operands[0], axis=int(params.get("axis", 1)))
    if op == "ak.concatenate":
        return ak.concatenate(list(operands), axis=int(params.get("axis", 1)))
    if op == "ak.flatten":
        return ak.flatten(operands[0], axis=_axis(params, default=1))
    if op == "ak.fill_none":
        return ak.fill_none(operands[0], params["value"], axis=int(params.get("axis", -1)))
    if op == "ak.drop_none":
        return ak.drop_none(operands[0])
    if op == "ak.where":
        cond, a, b = _three_operands(operands, params)
        return ak.where(cond, a, b)
    if op == "ak.zeros_like":
        return ak.zeros_like(operands[0], dtype=_dtype(params))
    if op == "ak.ones_like":
        return ak.ones_like(operands[0], dtype=_dtype(params))
    if op == "ak.values_astype":
        return ak.values_astype(operands[0], _dtype(params))
    if op == "ak.sort":
        return ak.sort(
            operands[0], axis=int(params.get("axis", 1)), ascending=bool(params.get("ascending", True))
        )
    if op == "ak.ravel":
        return ak.ravel(operands[0])
    if op == "ak.run_lengths":
        return ak.run_lengths(operands[0])
    if op == "ak.mask":
        return ak.mask(operands[0], operands[1], valid_when=bool(params.get("valid_when", True)))
    if op == "ak.is_none":
        return ak.is_none(operands[0], axis=int(params.get("axis", 0)))
    if op == "ak.singletons":
        return ak.singletons(operands[0], axis=int(params.get("axis", 0)))
    if op == "ak.pad_none":
        return ak.pad_none(
            operands[0],
            int(params["target"]),
            axis=int(params.get("axis", 1)),
            clip=bool(params.get("clip", False)),
        )
    if op == "ak.unflatten":
        return ak.unflatten(operands[0], operands[1], axis=int(params.get("axis", 0)))
    if op == "ak.to_regular":
        return ak.to_regular(operands[0], axis=int(params.get("axis", 1)))
    if op == "ak.from_regular":
        return ak.from_regular(operands[0], axis=int(params.get("axis", 1)))
    if op == "ak.full_like":
        dtype = np.dtype(str(params["dtype"])) if "dtype" in params else None
        return ak.full_like(operands[0], params["value"], dtype=dtype)
    if op == "ak.nan_to_num":
        return ak.nan_to_num(operands[0])
    if op == "ak.isclose":
        return ak.isclose(
            operands[0],
            operands[1],
            rtol=float(params.get("rtol", 1e-05)),
            atol=float(params.get("atol", 1e-08)),
        )
    if op == "ak.argcombinations":
        return ak.argcombinations(operands[0], int(params["n"]), fields=_fields(params) or None)
    if op == "ak.argcartesian":
        return ak.argcartesian(list(operands), nested=bool(params.get("nested", False)))
    if op == "ak.without_field":
        return ak.without_field(operands[0], params["field"])
    if op == "ak.broadcast_arrays":
        return ak.broadcast_arrays(*operands)[int(params["index"])]
    if op == "fields":
        names = [f for f in str(params["fields"]).split(",") if f]
        return operands[0][names]
    raise TypeError(f"unsupported awkward op {op!r}")


def _global_extremum(op: str, x: Any) -> Any:
    """axis=None min/max/ptp need typetracer-aware handling: the typetracer's option-scalar
    (MaybeNone) cannot be formed or subtracted, and upstream ak.ptp(axis=None) indexes its scalar
    result. On the TYPETRACER path use awkward's own mask_identity=False inference (a plain
    unknown scalar); the REAL path keeps awkward's default semantics untouched. ptp composes
    max - min on the typetracer (identical kernels — not hand-rolled inference)."""
    tracing = ak.backend(x) == "typetracer"
    if op == "ak.ptp":
        if tracing:
            high = ak.max(x, axis=None, mask_identity=False)
            low = ak.min(x, axis=None, mask_identity=False)
            return high - low
        return ak.ptp(x, axis=None)
    fn = ak.min if op == "ak.min" else ak.max
    return fn(x, axis=None, mask_identity=False) if tracing else fn(x, axis=None)


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
        out.append(params[key] if key in params else next(it))
    return out[0], out[1], out[2]
