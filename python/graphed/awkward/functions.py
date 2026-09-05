"""The ``gak`` namespace: awkward-style functions that record graphed nodes (plan M3).

Mirrors the subset of the awkward API the corpus analyses use, so an analysis written against
``gak`` records a backend-agnostic graph (the AwkwardBackend infers forms via the typetracer).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, SupportsFloat

from graphed import Array, ParamValue

from . import payloads


def join(left: Array, right: Array, *, on: Sequence[str], how: str = "inner", grouped: bool = False) -> Array:
    """The awkward-only ``gak.join`` verb (plan §3.1, a4). ``grouped=False`` is a thin alias to the
    neutral :func:`graphed.join` (relational, SQL-*duplicating*). ``grouped=True`` adds the awkward-only
    convenience: the relational result regrouped by a deterministic ``ak.unflatten`` — one sublist per
    matching build row — flattening back to the relational result. The grouped shape lives ONLY here
    (numpy has no ``gak``), so the neutral ``graphed.join`` rejects ``grouped`` outright."""
    # local imports: the neutral shuffle/join surface, kept off module-load to avoid an import cycle.
    import graphed  # noqa: PLC0415
    from graphed.errors import GraphedError  # noqa: PLC0415
    from graphed.shuffle import JOINKEY  # noqa: PLC0415

    if not grouped:
        return graphed.join(left, right, on=on, how=how)
    if left.session is not right.session:
        raise GraphedError("gak.join: left and right must belong to the same Session")
    session = left.session
    lk = graphed.pack_key(left, on=on)
    rk = graphed.pack_key(right, on=on)
    le = session.record_exchange(lk, {"scheme": "hash", "key": JOINKEY})
    re = session.record_exchange(rk, {"scheme": "hash", "key": JOINKEY})
    # same co-partitioning + full-key coalescing as graphed.join; grouped flips the Join to the regroup post-op.
    return session.record_join(le, re, {"on": ",".join([*on, JOINKEY]), "how": how, "grouped": True})


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


# ---- M28: variadic call templates for the recorded externals --------------------------------------
# The template ("$i" slots, [..] groups, constants where the callee allows) is PRESERVED NODE
# CONTENT: it rides the IR as canonical JSON, and record-time evaluation materializes THE SAME
# template the preservation replay obeys — agreement by construction. Descriptors on this path
# use the content-identity hash conventions shared with graphed-preserve, and params carry no
# filesystem paths. The template-less calls below remain the M3 path, byte-for-byte.


def _payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    from pathlib import Path  # noqa: PLC0415

    return Path(payload).read_bytes()


_TemplateEntry = tuple[str, Any]  # ("slot", i) | ("group", [i, ...]) | ("const", value)


def _parse_template(spec: object, n_inputs: int, *, constants: bool, groups: bool) -> list[_TemplateEntry]:
    if not isinstance(spec, list):
        raise TypeError("a call template's args must be a list (use kwargs= for keywords)")
    entries: list[_TemplateEntry] = []
    for entry in spec:
        if isinstance(entry, str) and entry.startswith("$"):
            i = int(entry[1:])
            if not 0 <= i < n_inputs:
                raise ValueError(f"call template slot {entry} out of range ({n_inputs} inputs)")
            entries.append(("slot", i))
        elif isinstance(entry, list):
            if not groups:
                raise TypeError("this callee takes scalar/array arguments, not stacked groups")
            entries.append(("group", [int(e[1:]) for e in entry]))
        elif constants and isinstance(entry, (str, int, float, bool)):
            entries.append(("const", entry))
        else:
            raise TypeError(f"call template entry {entry!r} is not a slot, group, or allowed constant")
    return entries


def _ml_matrix(entry: _TemplateEntry, vals: Sequence[object]) -> object:
    import awkward as _ak  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415

    kind, value = entry
    idxs = [value] if kind == "slot" else list(value)
    cols = [_np.asarray(_ak.to_numpy(_ak.Array(vals[i])), dtype="float32") for i in idxs]
    return _np.stack(cols, axis=1)


@dataclass
class _TemplateExternal:
    """The template path's recorded evaluator — PICKLABLE, unlike the closure it replaced.

    In-process it calls the recording-time ``call`` (the user's ``evaluator``/``runner``, routed
    through the template). A stdlib pickle — a plan shipped to a process pool, or a checkpointed
    plan — DROPS that callable, because neither a closure nor a live correctionlib/onnxruntime
    handle survives one; the worker rebuilds the resource from ``payload`` through the matching
    preserve plugin, which obeys the same template and the same content identity. So a ``call``
    that is NOT the plugin's own evaluation (an ``evaluator`` that scales or wraps it) diverges
    silently between in-process and out-of-process backends — for a backend-agnostic plan, ``call``
    must be the plugin's evaluation. No cloudpickle: these payloads are preservable, not opaque
    (§A.3.1)."""

    kind: str
    payload: bytes
    params: dict[str, str]
    call: Callable[..., object] | None

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "call": None}

    def __call__(self, *values: object) -> object:
        if self.call is None:
            from ..preserve.externals import get_plugin  # noqa: PLC0415  (import cycle)
            from ..preserve.externals._base import _PluginEvaluator  # noqa: PLC0415

            plugin = get_plugin(self.kind)
            if plugin is None:
                raise RuntimeError(
                    f"external payload kind {self.kind!r} has no registered plugin, so a worker "
                    "cannot rebuild it from the payload bytes"
                )
            # _PluginEvaluator keys the per-process resource cache on (kind, content_hash), so the
            # correction set / inference session is built once per worker, not once per partition.
            self.call = _PluginEvaluator(plugin, self.payload, self.params)
        return self.call(*values)


def apply_correction(
    payload: str | bytes,
    name: str,
    inputs: Sequence[Array],
    evaluator: Callable[..., object],
    *,
    args: list[object] | None = None,
) -> Array:
    """Record a correctionlib scale-factor application as an External node.

    With ``args=`` (e.g. ``["nominal", "$0", "$1"]``): the M28 preservation-aligned path —
    content-identity descriptor, no path in the IR, inputs passed to ``evaluator`` NATIVELY
    (awkward/numpy, jagged preserved) per the template, which replay obeys identically.
    Without it: the original M3 recording, unchanged (``payload`` must then be a path).

    Out-of-process backends (checkpoint, process pool) re-evaluate the correction canonically from
    the payload; ``evaluator`` is only the in-process eager path, so pass the plugin's own
    evaluation (e.g. ``cset[name].evaluate``) — a wrapper that alters it will diverge by backend."""
    if args is None:
        return inputs[0].session.record_external(
            "correction", evaluator, list(inputs), {"path": str(payload), "name": name}
        )
    blob = _payload_bytes(payload)
    entries = _parse_template(args, len(inputs), constants=True, groups=False)
    first_slot = next(int(i) for kind, i in entries if kind == "slot")

    def _fn(*vals: object) -> object:
        call = [vals[int(v)] if kind == "slot" else v for kind, v in entries]
        return evaluator(*call)

    session = inputs[0].session
    descriptor = payloads.correctionlib_contents_descriptor(blob, name)
    params = {"name": name, "args": json.dumps(args, sort_keys=False)}
    return session.record_external(
        "correction",
        _TemplateExternal(descriptor.kind, blob, {**params, "content_hash": descriptor.content_hash}, _fn),
        list(inputs),
        params,
        descriptor=descriptor,
        form=session.form(inputs[first_slot]),
    )


def onnx_inference(
    payload: str | bytes,
    inputs: Sequence[Array],
    runner: Callable[..., object],
    *,
    args: list[object] | None = None,
    kwargs: dict[str, object] | None = None,
) -> Array:
    """Record an ONNX model evaluation as an External node.

    With ``args=``/``kwargs=`` (e.g. ``args=[["$0", "$1"]]``): the M28 preservation-aligned
    path — weights-identity descriptor, no path in the IR, template entries materialized as
    float32 feature matrices (groups stack) and passed to ``runner`` positionally/by keyword.
    Without them: the original M3 recording, unchanged (``payload`` must then be a path).

    Out-of-process backends re-run the model canonically from the payload; ``runner`` is only the
    in-process eager path, so it must be the plugin's own inference (a wrapper diverges by backend)."""
    if args is None and kwargs is None:
        return inputs[0].session.record_external("onnx", runner, list(inputs), {"path": str(payload)})
    blob = _payload_bytes(payload)
    arg_entries = _parse_template(args or [], len(inputs), constants=False, groups=True)
    kw_entries = {
        k: _parse_template([v], len(inputs), constants=False, groups=True)[0]
        for k, v in (kwargs or {}).items()
    }
    slot_indices = [i for kind, i in arg_entries if kind == "slot"]
    group_first = [v[0] for kind, v in arg_entries if kind == "group"]
    first = (slot_indices + group_first + [0])[0]

    def _fn(*vals: object) -> object:
        call = [_ml_matrix(e, vals) for e in arg_entries]
        kw = {k: _ml_matrix(e, vals) for k, e in kw_entries.items()}
        return runner(*call, **kw)

    session = inputs[0].session
    params: dict[str, str] = {}
    if args is not None:
        params["args"] = json.dumps(args, sort_keys=False)
    if kwargs is not None:
        params["kwargs"] = json.dumps(kwargs, sort_keys=True)
    descriptor = payloads.onnx_weights_descriptor(blob)
    return session.record_external(
        "onnx",
        _TemplateExternal(descriptor.kind, blob, {**params, "content_hash": descriptor.content_hash}, _fn),
        list(inputs),
        params,
        descriptor=descriptor,
        form=session.form(inputs[first]),
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


def unflatten(arr: Array, counts: Array | int, axis: int = 0) -> Array:
    """``ak.unflatten``'s counts is "an integer or a one-dimensional array of integers"; an integer
    is a PARAMETER, not an input, so it never enters the graph as an edge."""
    if isinstance(counts, int):
        return arr.session.record_op("ak.unflatten", [arr], {"axis": axis, "counts": counts})
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


# ---- §2.3c: the gak dispatch layer and its per-function classification ----------------------
# An UNDISPOSED function does not fail loudly — `Varied`'s label-mapping `getattr` turns an
# unhandled duck-typed read into a recorded op and silently compiles nonsense — so the table below
# is exhaustive over this module's public surface and a frozen gate enumerates it dynamically. A
# new gak function is fixed HERE, never by editing that test.
#
#   *broadcast*             elementwise/structural, one recorded call per universe
#   *container-traversing*  the arrays live INSIDE a Mapping/Sequence argument
#   *tuple-returning*       the wrapper rebuilds its result, so it emits one Varied per position
#   *eager-metadata*        answers on the nominal member (sound by §2.1's form compatibility)
#   *refusing*              §5.4 boundary verbs
_GAK_EAGER_METADATA = ("backend_of", "fields", "head", "sample", "to_list", "type_of")
_GAK_REFUSING = ("join",)
_GAK_TUPLE_RETURNING = ("broadcast_arrays", "unzip")
_GAK_CONTAINER_TRAVERSING = (
    "apply_correction", "argcartesian", "cartesian", "concatenate", "onnx_inference", "zip",
)  # fmt: skip
_GAK_BROADCAST = (
    "all", "any", "argcombinations", "argmax", "argmin", "argsort", "combinations", "corr",
    "count", "count_nonzero", "covar", "drop_none", "fill_none", "firsts", "flatten",
    "from_regular", "full_like", "is_none", "isclose", "linear_fit", "local_index", "mask", "max",
    "mean", "min", "moment", "nan_to_num", "num", "ones_like", "pad_none", "prod", "ptp", "ravel",
    "run_lengths", "singletons", "softmax", "sort", "std", "sum", "to_regular", "unflatten",
    "values_astype", "var", "where", "with_field", "with_name", "with_parameter", "without_field",
    "without_parameters", "zeros_like",
)  # fmt: skip

GAK_DISPOSITIONS: dict[str, str] = {
    **dict.fromkeys(_GAK_BROADCAST, "broadcast"),
    **dict.fromkeys(_GAK_CONTAINER_TRAVERSING, "container-traversing"),
    **dict.fromkeys(_GAK_TUPLE_RETURNING, "tuple-returning"),
    **dict.fromkeys(_GAK_EAGER_METADATA, "eager-metadata"),
    **dict.fromkeys(_GAK_REFUSING, "refusing"),
}


@dataclass(frozen=True)
class GakSlot:
    """A substitution SLOT in an argument fixture, naming the operand KIND it needs.

    graphed type-checks the primary operand at RECORD time through the backend's `op_form`, so one
    array cannot serve every function; §2.3e's propagation gate owns one contexted array per kind
    and substitutes the one a slot names. The vocabulary is frozen at m48: flat numeric, jagged
    numeric, record, boolean mask, option type.
    """

    kind: str


@dataclass(frozen=True)
class GakArgFixture:
    """The AUXILIARY arguments §2.3e's propagation gate needs to call one function.

    It lives here beside the classification, not in the frozen test, so a new gak function arrives
    with both and the test stays untouched. The CONTEXTED operand is the test's, substituted into
    the slots below — a fixture-supplied context-free primary would degrade the check to
    `None == None`.
    """

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def _echo(*values: object, **_kwargs: object) -> object:
    """A stand-in payload evaluator for the two External-recording fixtures below."""
    return values[0]


_FLAT = GakSlot("flat")
_JAGGED = GakSlot("jagged")
_RECORD = GakSlot("record")
_MASK = GakSlot("mask")
_OPTION = GakSlot("option")
_INNER = {"axis": 1}

GAK_ARG_FIXTURES: dict[str, GakArgFixture] = {
    "all": GakArgFixture((_MASK,), _INNER),
    "any": GakArgFixture((_MASK,), _INNER),
    "apply_correction": GakArgFixture(
        (b'{"schema_version":2,"corrections":[]}', "sf", [_FLAT], _echo), {"args": ["$0"]}
    ),
    "argcartesian": GakArgFixture(([_JAGGED, _JAGGED],)),
    "argcombinations": GakArgFixture((_JAGGED, 2)),
    "argmax": GakArgFixture((_JAGGED,), _INNER),
    "argmin": GakArgFixture((_JAGGED,), _INNER),
    "argsort": GakArgFixture((_JAGGED,)),
    "broadcast_arrays": GakArgFixture((_FLAT, _JAGGED)),
    "cartesian": GakArgFixture(([_JAGGED, _JAGGED],)),
    "combinations": GakArgFixture((_JAGGED, 2)),
    "concatenate": GakArgFixture(([_FLAT, _FLAT],)),
    "corr": GakArgFixture((_JAGGED, _JAGGED), _INNER),
    "count": GakArgFixture((_JAGGED,), _INNER),
    "count_nonzero": GakArgFixture((_JAGGED,), _INNER),
    "covar": GakArgFixture((_JAGGED, _JAGGED), _INNER),
    "drop_none": GakArgFixture((_OPTION,)),
    "fill_none": GakArgFixture((_OPTION, 0.0)),
    "firsts": GakArgFixture((_JAGGED,)),
    "flatten": GakArgFixture((_JAGGED,)),
    "from_regular": GakArgFixture((_JAGGED,)),
    "full_like": GakArgFixture((_JAGGED, 1.0)),
    "is_none": GakArgFixture((_OPTION,)),
    "isclose": GakArgFixture((_JAGGED, _JAGGED)),
    "linear_fit": GakArgFixture((_JAGGED, _JAGGED), _INNER),
    "local_index": GakArgFixture((_JAGGED,)),
    "mask": GakArgFixture((_JAGGED, _MASK)),
    "max": GakArgFixture((_JAGGED,), _INNER),
    "mean": GakArgFixture((_JAGGED,), _INNER),
    "min": GakArgFixture((_JAGGED,), _INNER),
    "moment": GakArgFixture((_JAGGED, 2), _INNER),
    "nan_to_num": GakArgFixture((_JAGGED,)),
    "num": GakArgFixture((_JAGGED,)),
    "ones_like": GakArgFixture((_JAGGED,)),
    # b"" is a valid (empty) ModelProto: the weights-identity path needs a PARSEABLE model, not a
    # real one, and the path-based form would need a file on disk.
    "onnx_inference": GakArgFixture((b"", [_FLAT], _echo), {"args": [["$0"]]}),
    "pad_none": GakArgFixture((_JAGGED, 2)),
    "prod": GakArgFixture((_JAGGED,), _INNER),
    "ptp": GakArgFixture((_JAGGED,), _INNER),
    "ravel": GakArgFixture((_JAGGED,)),
    "run_lengths": GakArgFixture((_FLAT,)),
    "singletons": GakArgFixture((_OPTION,)),
    "softmax": GakArgFixture((_JAGGED,)),
    "sort": GakArgFixture((_JAGGED,)),
    "std": GakArgFixture((_JAGGED,), _INNER),
    "sum": GakArgFixture((_JAGGED,), _INNER),
    "to_regular": GakArgFixture((_JAGGED,)),
    "unflatten": GakArgFixture((_JAGGED, 2), _INNER),
    "unzip": GakArgFixture((_RECORD,)),
    "values_astype": GakArgFixture((_JAGGED, "float32")),
    "var": GakArgFixture((_JAGGED,), _INNER),
    "where": GakArgFixture((_MASK, 1.0, 0.0)),
    "with_field": GakArgFixture((_RECORD, _JAGGED, "extra")),
    "with_name": GakArgFixture((_RECORD, "Thing")),
    "with_parameter": GakArgFixture((_RECORD, "key", "value")),
    "without_field": GakArgFixture((_RECORD, "pt")),
    "without_parameters": GakArgFixture((_RECORD,)),
    "zeros_like": GakArgFixture((_JAGGED,)),
    "zip": GakArgFixture(({"pt": _JAGGED, "eta": _JAGGED},)),
}


def _dispatched(fn: Callable[..., Any], kind: str) -> Callable[..., Any]:
    """Wrap one gak function with the dispatch its classification names (§2.3c).

    Signatures do not change (R17.0's anti-drift pin): `functools.wraps` keeps `__wrapped__`, so
    `inspect.signature` still reports the underlying one.
    """
    from graphed.varied import (  # noqa: PLC0415
        boundary_refusal,
        containers_in,
        expand,
        expand_tuple,
        narrow,
    )

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not containers_in(*args, *kwargs.values()):
            return fn(*args, **kwargs)
        if kind == "refusing":
            raise boundary_refusal(f"gak.{fn.__name__}", *args, *kwargs.values())
        if kind == "eager-metadata":
            return fn(
                *(narrow(arg, "nominal") for arg in args),
                **{key: narrow(value, "nominal") for key, value in kwargs.items()},
            )
        if kind == "tuple-returning":
            return expand_tuple(fn, args, kwargs)
        return expand(fn, args, kwargs)

    return wrapper


for _gak_name, _gak_kind in GAK_DISPOSITIONS.items():
    globals()[_gak_name] = _dispatched(globals()[_gak_name], _gak_kind)
