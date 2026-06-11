"""The ``gak`` namespace: awkward-style functions that record graphed nodes (plan M3).

Mirrors the subset of the awkward API the corpus analyses use, so an analysis written against
``gak`` records a backend-agnostic graph (the AwkwardBackend infers forms via the typetracer).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import SupportsFloat

from graphed import Array, ParamValue

from . import payloads


def _comb_params(
    n: int,
    replacement: bool,
    axis: int,
    fields: Sequence[str] | None,
    with_name: str | None,
    parameters: Mapping[str, ParamValue] | None,
) -> dict[str, ParamValue]:
    params: dict[str, ParamValue] = {"n": n}
    if fields:
        params["fields"] = ",".join(fields)
    if replacement:
        params["replacement"] = True
    if axis != 1:
        params["axis"] = axis
    params.update(_structure_params(with_name=with_name, parameters=parameters))
    return params


def combinations(
    arr: Array,
    n: int,
    *,
    replacement: bool = False,
    axis: int = 1,
    fields: Sequence[str] | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    return arr.session.record_op(
        "ak.combinations", [arr], _comb_params(n, replacement, axis, fields, with_name, parameters)
    )


def _cartesian_params(
    axis: int,
    nested: bool | Sequence[int] | None,
    with_name: str | None,
    parameters: Mapping[str, ParamValue] | None,
) -> dict[str, ParamValue]:
    params: dict[str, ParamValue] = {}
    if isinstance(nested, bool):
        params["nested"] = nested
    elif nested is not None:
        params["nested_axes"] = ",".join(str(int(i)) for i in nested)
    if axis != 1:
        params["axis"] = axis
    params.update(_structure_params(with_name=with_name, parameters=parameters))
    return params


def cartesian(
    arrays: Sequence[Array],
    *,
    axis: int = 1,
    nested: bool | Sequence[int] | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    return arrays[0].session.record_op(
        "ak.cartesian", list(arrays), _cartesian_params(axis, nested, with_name, parameters)
    )


def _structure_params(
    *, with_name: str | None, parameters: Mapping[str, ParamValue] | None
) -> dict[str, ParamValue]:
    """The shared inline record-naming/parameter kwargs (present-only; JSON for the dict)."""
    out: dict[str, ParamValue] = {}
    if with_name is not None:
        out["with_name"] = with_name
    if parameters:
        out["parameters"] = json.dumps(dict(parameters), sort_keys=True)
    return out


def zip(
    fields: Mapping[str, Array] | Sequence[Array],
    *,
    depth_limit: int | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    """``ak.zip`` parity: a mapping makes named records, a SEQUENCE makes tuple records
    (fields "0", "1", ...); ``depth_limit=1`` builds an events-level record of collections
    without broadcasting their jaggedness together."""
    if isinstance(fields, Mapping):
        arrays = list(fields.values())
        params: dict[str, ParamValue] = {"fields": ",".join(fields.keys())}
    else:
        arrays = list(fields)
        params = {"tuple": True}
    if depth_limit is not None:
        params["depth_limit"] = depth_limit
    params.update(_structure_params(with_name=with_name, parameters=parameters))
    return arrays[0].session.record_op("ak.zip", arrays, params)


def with_field(arr: Array, value: Array, where: str) -> Array:
    return arr.session.record_op("ak.with_field", [arr, value], {"field": where})


def num(arr: Array, axis: int = 1) -> Array:
    return arr.session.record_op("ak.num", [arr], {"axis": axis})


def count(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    return _reduce("ak.count", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def _reduce(
    name: str,
    arr: Array,
    axis: int | None,
    extra: Mapping[str, int] | None = None,
    *,
    keepdims: bool = False,
    mask_identity: bool | None = None,
    initial: float | None = None,
    weight: Array | None = None,
) -> Array:
    """Record one awkward reduction under the M12/M16 STRUCTURAL RULE: reducing over the event
    (partitioned) axis — axis None or 0 — is a stage boundary executed by the M7 tree reduction;
    an inner-axis (per-event) reduction is partition-local and fusible. The ak parity kwargs are
    recorded PRESENT-ONLY (awkward's own defaults apply when absent); a weight is a second
    INPUT, not a parameter."""
    params: dict[str, object] = {"axis": "none" if axis is None else axis}
    if extra:
        params.update(extra)
    if keepdims:
        params["keepdims"] = True
    if mask_identity is not None:
        params["mask_identity"] = mask_identity
    if initial is not None:
        params["initial"] = float(initial)
    inputs = [arr] if weight is None else [arr, weight]
    if weight is not None:
        params["weighted"] = True
    boundary = axis is None or axis == 0
    return arr.session.record_op(name, inputs, params, reduction=boundary)  # type: ignore[arg-type]


def sum(arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False) -> Array:
    return _reduce("ak.sum", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def any(arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False) -> Array:
    return _reduce("ak.any", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def all(arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False) -> Array:
    return _reduce("ak.all", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def count_nonzero(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    return _reduce("ak.count_nonzero", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def min(
    arr: Array,
    axis: int | None = None,
    *,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
) -> Array:
    return _reduce(
        "ak.min",
        arr,
        axis,
        keepdims=keepdims,
        initial=initial,
        mask_identity=None if mask_identity else False,
    )


def max(
    arr: Array,
    axis: int | None = None,
    *,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
) -> Array:
    return _reduce(
        "ak.max",
        arr,
        axis,
        keepdims=keepdims,
        initial=initial,
        mask_identity=None if mask_identity else False,
    )


def prod(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    return _reduce("ak.prod", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def mean(
    arr: Array,
    axis: int | None = None,
    *,
    weight: Array | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Array:
    return _reduce(
        "ak.mean",
        arr,
        axis,
        weight=weight,
        keepdims=keepdims,
        mask_identity=mask_identity or None,
    )


def ptp(arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = True) -> Array:
    return _reduce("ak.ptp", arr, axis, keepdims=keepdims, mask_identity=None if mask_identity else False)


def std(
    arr: Array,
    axis: int | None = None,
    *,
    ddof: int = 0,
    weight: Array | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Array:
    return _reduce(
        "ak.std",
        arr,
        axis,
        {"ddof": ddof} if ddof else None,
        weight=weight,
        keepdims=keepdims,
        mask_identity=mask_identity or None,
    )


def var(
    arr: Array,
    axis: int | None = None,
    *,
    ddof: int = 0,
    weight: Array | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Array:
    return _reduce(
        "ak.var",
        arr,
        axis,
        {"ddof": ddof} if ddof else None,
        weight=weight,
        keepdims=keepdims,
        mask_identity=mask_identity or None,
    )


def moment(
    arr: Array,
    n: int,
    axis: int | None = None,
    *,
    weight: Array | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Array:
    return _reduce(
        "ak.moment",
        arr,
        axis,
        {"n": n},
        weight=weight,
        keepdims=keepdims,
        mask_identity=mask_identity or None,
    )


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


def argmin(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = True
) -> Array:
    params: dict[str, ParamValue] = {"axis": "none" if axis is None else axis, "keepdims": keepdims}
    if not mask_identity:
        params["mask_identity"] = False
    return arr.session.record_op("ak.argmin", [arr], params)


def argmax(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = True
) -> Array:
    params: dict[str, ParamValue] = {"axis": "none" if axis is None else axis, "keepdims": keepdims}
    if not mask_identity:
        params["mask_identity"] = False
    return arr.session.record_op("ak.argmax", [arr], params)


def argsort(arr: Array, axis: int = -1, *, ascending: bool = True, stable: bool = True) -> Array:
    params: dict[str, ParamValue] = {"axis": axis, "ascending": ascending}
    if not stable:
        params["stable"] = False
    return arr.session.record_op("ak.argsort", [arr], params)


def local_index(arr: Array, axis: int = -1) -> Array:
    return arr.session.record_op("ak.local_index", [arr], {"axis": axis})


def concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
    return arrays[0].session.record_op("ak.concatenate", list(arrays), {"axis": axis})


def flatten(arr: Array, axis: int | None = 1) -> Array:
    return arr.session.record_op("ak.flatten", [arr], {"axis": "none" if axis is None else axis})


def fill_none(arr: Array, value: bool | int | float, axis: int = -1) -> Array:
    return arr.session.record_op("ak.fill_none", [arr], {"value": value, "axis": axis})


def drop_none(arr: Array, axis: int | None = None) -> Array:
    params: dict[str, ParamValue] = {} if axis is None else {"axis": axis}
    return arr.session.record_op("ak.drop_none", [arr], params)


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


def zeros_like(arr: Array, *, dtype: str | None = None) -> Array:
    """``ak.zeros_like`` parity: ``dtype=None`` PRESERVES the input dtype (no forced int64)."""
    params: dict[str, ParamValue] = {} if dtype is None else {"dtype": dtype}
    return arr.session.record_op("ak.zeros_like", [arr], params)


def ones_like(arr: Array, *, dtype: str | None = None) -> Array:
    params: dict[str, ParamValue] = {} if dtype is None else {"dtype": dtype}
    return arr.session.record_op("ak.ones_like", [arr], params)


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


# ---- M17: structure-op parity (dask-awkward parity plan P1) ---------------------------------
def sort(arr: Array, axis: int = -1, *, ascending: bool = True, stable: bool = True) -> Array:
    params: dict[str, ParamValue] = {"axis": axis, "ascending": ascending}
    if not stable:
        params["stable"] = False
    return arr.session.record_op("ak.sort", [arr], params)


def ravel(arr: Array) -> Array:
    return arr.session.record_op("ak.ravel", [arr])


def run_lengths(arr: Array) -> Array:
    return arr.session.record_op("ak.run_lengths", [arr])


def mask(arr: Array, condition: Array, *, valid_when: bool = True) -> Array:
    return arr.session.record_op("ak.mask", [arr, condition], {"valid_when": valid_when})


def is_none(arr: Array, axis: int = 0) -> Array:
    return arr.session.record_op("ak.is_none", [arr], {"axis": axis})


def singletons(arr: Array, axis: int = 0) -> Array:
    return arr.session.record_op("ak.singletons", [arr], {"axis": axis})


def pad_none(arr: Array, target: int, axis: int = 1, *, clip: bool = False) -> Array:
    return arr.session.record_op("ak.pad_none", [arr], {"target": target, "axis": axis, "clip": clip})


def unflatten(arr: Array, counts: Array, axis: int = 0) -> Array:
    return arr.session.record_op("ak.unflatten", [arr, counts], {"axis": axis})


def to_regular(arr: Array, axis: int = 1) -> Array:
    return arr.session.record_op("ak.to_regular", [arr], {"axis": axis})


def from_regular(arr: Array, axis: int = 1) -> Array:
    return arr.session.record_op("ak.from_regular", [arr], {"axis": axis})


def full_like(arr: Array, value: float, *, dtype: str | None = None) -> Array:
    params: dict[str, object] = {"value": float(value)}
    if dtype is not None:
        params["dtype"] = dtype
    return arr.session.record_op("ak.full_like", [arr], params)  # type: ignore[arg-type]


def nan_to_num(
    arr: Array, *, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None
) -> Array:
    params: dict[str, ParamValue] = {}
    if nan != 0.0:
        params["nan"] = nan
    if posinf is not None:
        params["posinf"] = posinf
    if neginf is not None:
        params["neginf"] = neginf
    return arr.session.record_op("ak.nan_to_num", [arr], params)


def isclose(
    x: Array, y: Array, *, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> Array:
    params: dict[str, ParamValue] = {"rtol": rtol, "atol": atol}
    if equal_nan:
        params["equal_nan"] = True
    return x.session.record_op("ak.isclose", [x, y], params)


def argcombinations(
    arr: Array,
    n: int,
    *,
    replacement: bool = False,
    axis: int = 1,
    fields: Sequence[str] | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    return arr.session.record_op(
        "ak.argcombinations", [arr], _comb_params(n, replacement, axis, fields, with_name, parameters)
    )


def argcartesian(
    arrays: Sequence[Array],
    *,
    axis: int = 1,
    nested: bool | Sequence[int] | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    return arrays[0].session.record_op(
        "ak.argcartesian", list(arrays), _cartesian_params(axis, nested, with_name, parameters)
    )


def without_field(arr: Array, field: str) -> Array:
    return arr.session.record_op("ak.without_field", [arr], {"field": field})


def values_astype(arr: Array, dtype: str) -> Array:
    return arr.session.record_op("ak.values_astype", [arr], {"dtype": dtype})


def broadcast_arrays(*arrays: Array, depth_limit: int | None = None) -> tuple[Array, ...]:
    """Each broadcast output is its own recorded node (same inputs, an index param)."""
    session = arrays[0].session
    extra: dict[str, ParamValue] = {} if depth_limit is None else {"depth_limit": depth_limit}
    return tuple(
        session.record_op("ak.broadcast_arrays", list(arrays), {"index": i, **extra})
        for i in range(len(arrays))
    )


def unzip(arr: Array) -> tuple[Array, ...]:
    """One recorded field op per record field (the field list comes from the typetracer form)."""
    form = arr.session.form(arr)
    return tuple(arr[name] for name in form.tt.fields)  # type: ignore[attr-defined]


def to_list(arr: Array) -> list[object]:
    """EAGER sugar: materializes through the session, then ak.to_list (records nothing new)."""
    import awkward as _ak  # noqa: PLC0415  (avoid importing awkward at gak import for tooling)

    out: list[object] = _ak.to_list(arr.session.materialize(arr))
    return out


# ---- M18: behaviors (dask-awkward parity plan P2) ---------------------------------------------
def with_name(arr: Array, name: str) -> Array:
    """Name the records; with a behavior dict registered on the backend, behavior properties
    (vector's .pt/.mass) then work through plain attribute access."""
    return arr.session.record_op("ak.with_name", [arr], {"name": name})


def with_parameter(arr: Array, key: str, value: str | int | float | bool) -> Array:
    return arr.session.record_op("ak.with_parameter", [arr], {"key": key, "value": value})


def without_parameters(arr: Array) -> Array:
    return arr.session.record_op("ak.without_parameters", [arr])


# ---- M19: introspection + peeking conveniences (parity plan P3.8) ----------------------------
def fields(arr: Array) -> list[str]:
    """Record-field names from the FORM (pure metadata: records nothing)."""
    return list(arr.session.form(arr).tt.fields)  # type: ignore[attr-defined]


def type_of(arr: Array) -> str:
    """The awkward type string from the FORM (pure metadata: records nothing)."""
    return str(arr.session.form(arr).describe())


def backend_of(arr: Array) -> str:
    """The session's backend class name (pure metadata: records nothing)."""
    return type(arr.session.backend).__name__


def head(arr: Array, n: int = 5) -> object:
    """EAGER peek at the first ``n`` rows (the common slice op + reference materialize)."""
    return arr.session.materialize(arr[:n])


def sample(arr: Array, *, factor: int) -> object:
    """EAGER peek at every ``factor``-th row."""
    return arr.session.materialize(arr[::factor])
