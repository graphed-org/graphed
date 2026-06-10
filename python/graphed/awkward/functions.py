"""The ``gak`` namespace: awkward-style functions that record graphed nodes (plan M3).

Mirrors the subset of the awkward API the corpus analyses use, so an analysis written against
``gak`` records a backend-agnostic graph (the AwkwardBackend infers forms via the typetracer).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import SupportsFloat

from graphed import Array, ParamValue

from . import payloads


def combinations(arr: Array, n: int, *, fields: Sequence[str] | None = None) -> Array:
    params: dict[str, ParamValue] = {"n": n}
    if fields:
        params["fields"] = ",".join(fields)
    return arr.session.record_op("ak.combinations", [arr], params)


def cartesian(arrays: Sequence[Array], *, nested: bool = False) -> Array:
    return arrays[0].session.record_op("ak.cartesian", list(arrays), {"nested": nested})


def zip(fields: Mapping[str, Array]) -> Array:
    arrays = list(fields.values())
    return arrays[0].session.record_op("ak.zip", arrays, {"fields": ",".join(fields.keys())})


def with_field(arr: Array, value: Array, where: str) -> Array:
    return arr.session.record_op("ak.with_field", [arr, value], {"field": where})


def num(arr: Array, axis: int = 1) -> Array:
    return arr.session.record_op("ak.num", [arr], {"axis": axis})


def count(arr: Array, axis: int | None = 1) -> Array:
    return _reduce("ak.count", arr, axis)


def _reduce(name: str, arr: Array, axis: int | None, extra: Mapping[str, int] | None = None) -> Array:
    """Record one awkward reduction under the M12/M16 STRUCTURAL RULE: reducing over the event
    (partitioned) axis — axis None or 0 — is a stage boundary executed by the M7 tree reduction;
    an inner-axis (per-event) reduction is partition-local and fusible."""
    params: dict[str, object] = {"axis": "none" if axis is None else axis}
    if extra:
        params.update(extra)
    boundary = axis is None or axis == 0
    return arr.session.record_op(name, [arr], params, reduction=boundary)  # type: ignore[arg-type]


def sum(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.sum", arr, axis)


def any(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.any", arr, axis)


def all(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.all", arr, axis)


def count_nonzero(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.count_nonzero", arr, axis)


def min(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.min", arr, axis)


def max(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.max", arr, axis)


def prod(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.prod", arr, axis)


def mean(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.mean", arr, axis)


def ptp(arr: Array, axis: int | None = None) -> Array:
    return _reduce("ak.ptp", arr, axis)


def std(arr: Array, axis: int | None = None, *, ddof: int = 0) -> Array:
    return _reduce("ak.std", arr, axis, {"ddof": ddof} if ddof else None)


def var(arr: Array, axis: int | None = None, *, ddof: int = 0) -> Array:
    return _reduce("ak.var", arr, axis, {"ddof": ddof} if ddof else None)


def moment(arr: Array, n: int, axis: int | None = None) -> Array:
    return _reduce("ak.moment", arr, axis, {"n": n})


def softmax(arr: Array, axis: int = 1) -> Array:
    """Per-list normalization — shape-preserving, ALWAYS partition-local (never a boundary)."""
    return arr.session.record_op("ak.softmax", [arr], {"axis": axis})


def _reduce2(name: str, x: Array, y: Array, axis: int | None) -> Array:
    boundary = axis is None or axis == 0
    return x.session.record_op(name, [x, y], {"axis": "none" if axis is None else axis}, reduction=boundary)


def corr(x: Array, y: Array, axis: int | None = None) -> Array:
    return _reduce2("ak.corr", x, y, axis)


def covar(x: Array, y: Array, axis: int | None = None) -> Array:
    return _reduce2("ak.covar", x, y, axis)


def linear_fit(x: Array, y: Array, axis: int | None = None) -> Array:
    return _reduce2("ak.linear_fit", x, y, axis)


def firsts(arr: Array, axis: int = 1) -> Array:
    return arr.session.record_op("ak.firsts", [arr], {"axis": axis})


def argmin(arr: Array, axis: int = 1, *, keepdims: bool = False) -> Array:
    return arr.session.record_op("ak.argmin", [arr], {"axis": axis, "keepdims": keepdims})


def argmax(arr: Array, axis: int = 1, *, keepdims: bool = False) -> Array:
    return arr.session.record_op("ak.argmax", [arr], {"axis": axis, "keepdims": keepdims})


def argsort(arr: Array, axis: int = 1, *, ascending: bool = True) -> Array:
    return arr.session.record_op("ak.argsort", [arr], {"axis": axis, "ascending": ascending})


def local_index(arr: Array, axis: int = 1) -> Array:
    return arr.session.record_op("ak.local_index", [arr], {"axis": axis})


def concatenate(arrays: Sequence[Array], axis: int = 1) -> Array:
    return arrays[0].session.record_op("ak.concatenate", list(arrays), {"axis": axis})


def flatten(arr: Array, axis: int | None = 1) -> Array:
    return arr.session.record_op("ak.flatten", [arr], {"axis": "none" if axis is None else axis})


def fill_none(arr: Array, value: bool | int | float, axis: int = -1) -> Array:
    return arr.session.record_op("ak.fill_none", [arr], {"value": value, "axis": axis})


def drop_none(arr: Array) -> Array:
    return arr.session.record_op("ak.drop_none", [arr])


def where(cond: Array, a: object, b: object) -> Array:
    inputs: list[Array] = []
    params: dict[str, ParamValue] = {}
    for i, operand in enumerate((cond, a, b)):
        if isinstance(operand, Array):
            inputs.append(operand)
        elif isinstance(operand, bool | int | str):
            params[f"const{i}"] = operand
        elif isinstance(operand, SupportsFloat):
            params[f"const{i}"] = float(operand)
        else:
            raise TypeError(f"unsupported where operand {operand!r}")
    return cond.session.record_op("ak.where", inputs, params)


def zeros_like(arr: Array, *, dtype: str = "int64") -> Array:
    return arr.session.record_op("ak.zeros_like", [arr], {"dtype": dtype})


def ones_like(arr: Array, *, dtype: str = "int64") -> Array:
    return arr.session.record_op("ak.ones_like", [arr], {"dtype": dtype})


def apply_correction(
    json_path: str, name: str, inputs: Sequence[Array], evaluator: Callable[..., object]
) -> Array:
    """Record a correctionlib scale-factor application as an External node (content-hashed JSON)."""
    return inputs[0].session.record_external(
        "correction", lambda *vals: evaluator(*vals), list(inputs), {"path": json_path, "name": name}
    )


def onnx_inference(model_path: str, inputs: Sequence[Array], runner: Callable[..., object]) -> Array:
    """Record an ONNX model evaluation as an External node (content-hashed model file)."""
    return inputs[0].session.record_external(
        "onnx", lambda *vals: runner(*vals), list(inputs), {"path": model_path}
    )


# expose the payload helpers for direct descriptor construction/inspection
correctionlib_descriptor = payloads.correctionlib_descriptor
onnx_descriptor = payloads.onnx_descriptor
dataset_descriptor = payloads.dataset_descriptor
