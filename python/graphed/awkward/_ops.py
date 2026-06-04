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
_UNARY: dict[str, Any] = {
    "abs": np.abs,
    "neg": np.negative,
    "invert": np.invert,
    "sqrt": np.sqrt,
    "cos": np.cos,
    "sin": np.sin,
    "cosh": np.cosh,
    "sinh": np.sinh,
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
    if op == "ak.sum":
        return ak.sum(operands[0], axis=_axis(params))
    if op == "ak.any":
        return ak.any(operands[0], axis=_axis(params))
    if op == "ak.all":
        return ak.all(operands[0], axis=_axis(params))
    if op == "ak.count":
        return ak.count(operands[0], axis=_axis(params))
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
    raise TypeError(f"unsupported awkward op {op!r}")


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
