"""The ``gak`` namespace: awkward-style functions that record graphed nodes (plan M3).

Mirrors the subset of the awkward API the corpus analyses use, so an analysis written against
``gak`` records a backend-agnostic graph (the AwkwardBackend infers forms via the typetracer).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import awkward as ak
import numpy as np
import numpy.typing as npt

import graphed
from graphed import Array, ParamValue
from graphed.array import scalar_params
from graphed.errors import GraphedError, GraphedTypeError
from graphed.provenance import capture
from graphed.shuffle import JOINKEY
from graphed.varied import boundary_refusal, containers_in, expand, expand_tuple, narrow

from . import payloads
from ._dispositions import GAK_ARG_FIXTURES as GAK_ARG_FIXTURES
from ._dispositions import GAK_DISPOSITIONS as GAK_DISPOSITIONS
from ._dispositions import GakArgFixture as GakArgFixture
from ._dispositions import GakSlot as GakSlot
from ._params import _cartesian_params, _comb_params, _reduce, _reduce2, _structure_params
from ._templates import _ml_matrix, _parse_template, _payload_bytes
from .backend import AwkwardForm, astype_form, broadcast_form


def join(
    left: Array, right: Array, *, on: Sequence[str], how: str = "inner", grouped: bool = False
) -> Array:
    """Return the awkward-only ``gak.join`` verb (plan §3.1, a4).

    ``grouped=False`` is a thin alias to the neutral :func:`graphed.join` (relational,
    SQL-*duplicating*). ``grouped=True`` adds the awkward-only convenience: the relational result
    regrouped by a deterministic ``ak.unflatten`` — one sublist per matching build row — flattening
    back to the relational result. The grouped shape lives ONLY here (numpy has no ``gak``), so the
    neutral ``graphed.join`` rejects ``grouped`` outright.
    """
    if not grouped:
        return graphed.join(left, right, on=on, how=how)
    if left.session is not right.session:
        raise GraphedError(
            "gak.join: left and right must belong to the same Session; record both into one (read "
            "the right side inside `with left.session:`, or pass the reader session=)"
        )
    session = left.session
    lk = graphed.pack_key(left, on=on)
    rk = graphed.pack_key(right, on=on)
    le = session.record_exchange(lk, {"scheme": "hash", "key": JOINKEY})
    re = session.record_exchange(rk, {"scheme": "hash", "key": JOINKEY})
    # same co-partitioning + full-key coalescing as graphed.join; grouped flips the Join to the
    # regroup post-op.
    return session.record_join(
        le, re, {"on": ",".join([*on, JOINKEY]), "how": how, "grouped": True}
    )


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
    """Record ``ak.combinations`` of ``n`` items per list."""
    return arr.session.record_op(
        "ak.combinations", [arr], _comb_params(n, replacement, axis, fields, with_name, parameters)
    )


def cartesian(
    arrays: Sequence[Array] | Mapping[str, Array],
    *,
    axis: int = 1,
    nested: bool | Sequence[int] | Sequence[str] | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    """Record ``ak.cartesian`` over ``arrays``."""
    inputs = list(arrays.values() if isinstance(arrays, Mapping) else arrays)
    return inputs[0].session.record_op(
        "ak.cartesian", inputs, _cartesian_params(arrays, axis, nested, with_name, parameters)
    )


def zip(
    fields: Mapping[str, Array] | Sequence[Array],
    *,
    depth_limit: int | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    """``ak.zip`` parity: a mapping makes named records, a SEQUENCE makes tuple records.

    Tuple fields are "0", "1", ...; ``depth_limit=1`` builds an events-level record of collections
    without broadcasting their jaggedness together.
    """
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


def with_field(arr: Array, value: object, where: str) -> Array:
    """Record ``ak.with_field``, setting field ``where`` to ``value``."""
    if isinstance(value, Array):
        return arr.session.record_op("ak.with_field", [arr, value], {"field": where})
    try:
        # a scalar, which ak broadcasts, travels as a param in its own dtype (M59)
        params = scalar_params(value)
    except (TypeError, ValueError) as err:
        raise GraphedTypeError(
            "with_field",
            capture(),
            "the value must be a deferred array or a bool/int/float/str scalar (numpy scalars "
            f"keep their dtype), not {type(value).__name__}",
        ) from err
    return arr.session.record_op("ak.with_field", [arr], {"field": where, **params})


def num(arr: Array, axis: int = 1) -> Array:
    # axis 0 counts the partitioned axis: a per-chunk partial, like any axis-0 reduction
    """Record ``ak.num`` along ``axis``."""
    return arr.session.record_op("ak.num", [arr], {"axis": axis}, reduction=axis == 0)


def count(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    """Record ``ak.count`` along ``axis``."""
    return _reduce("ak.count", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def sum(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    """Record ``ak.sum`` along ``axis``."""
    return _reduce("ak.sum", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def any(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    """Record ``ak.any`` along ``axis``."""
    return _reduce("ak.any", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def all(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    """Record ``ak.all`` along ``axis``."""
    return _reduce("ak.all", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def count_nonzero(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = False
) -> Array:
    """Record ``ak.count_nonzero`` along ``axis``."""
    return _reduce(
        "ak.count_nonzero", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None
    )


def min(
    arr: Array,
    axis: int | None = None,
    *,
    keepdims: bool = False,
    initial: float | None = None,
    mask_identity: bool = True,
) -> Array:
    """Record ``ak.min`` along ``axis``."""
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
    """Record ``ak.max`` along ``axis``."""
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
    """Record ``ak.prod`` along ``axis``."""
    return _reduce("ak.prod", arr, axis, keepdims=keepdims, mask_identity=mask_identity or None)


def mean(
    arr: Array,
    axis: int | None = None,
    *,
    weight: Array | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Array:
    """Record ``ak.mean`` along ``axis``, optionally weighted."""
    return _reduce(
        "ak.mean",
        arr,
        axis,
        weight=weight,
        keepdims=keepdims,
        mask_identity=mask_identity or None,
    )


def ptp(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = True
) -> Array:
    """Record ``ak.ptp`` along ``axis``."""
    return _reduce(
        "ak.ptp", arr, axis, keepdims=keepdims, mask_identity=None if mask_identity else False
    )


def std(
    arr: Array,
    axis: int | None = None,
    *,
    ddof: int = 0,
    weight: Array | None = None,
    keepdims: bool = False,
    mask_identity: bool = False,
) -> Array:
    """Record ``ak.std`` along ``axis``, optionally weighted."""
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
    """Record ``ak.var`` along ``axis``, optionally weighted."""
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
    """Record ``ak.moment`` of order ``n`` along ``axis``, optionally weighted."""
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


def corr(x: Array, y: Array, axis: int | None = None) -> Array:
    """Record ``ak.corr`` of ``x`` and ``y`` along ``axis``."""
    return _reduce2("ak.corr", x, y, axis)


def covar(x: Array, y: Array, axis: int | None = None) -> Array:
    """Record ``ak.covar`` of ``x`` and ``y`` along ``axis``."""
    return _reduce2("ak.covar", x, y, axis)


def linear_fit(x: Array, y: Array, axis: int | None = None) -> Array:
    """Record ``ak.linear_fit`` of ``y`` against ``x`` along ``axis``."""
    return _reduce2("ak.linear_fit", x, y, axis)


def firsts(arr: Array, axis: int = 1) -> Array:
    """Record ``ak.firsts`` along ``axis``."""
    return arr.session.record_op("ak.firsts", [arr], {"axis": axis})


def argmin(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = True
) -> Array:
    """Record ``ak.argmin`` along ``axis``."""
    params: dict[str, ParamValue] = {
        "axis": "none" if axis is None else axis,
        "keepdims": keepdims,
    }
    if not mask_identity:
        params["mask_identity"] = False
    return arr.session.record_op("ak.argmin", [arr], params)


def argmax(
    arr: Array, axis: int | None = None, *, keepdims: bool = False, mask_identity: bool = True
) -> Array:
    """Record ``ak.argmax`` along ``axis``."""
    params: dict[str, ParamValue] = {
        "axis": "none" if axis is None else axis,
        "keepdims": keepdims,
    }
    if not mask_identity:
        params["mask_identity"] = False
    return arr.session.record_op("ak.argmax", [arr], params)


def argsort(arr: Array, axis: int = -1, *, ascending: bool = True, stable: bool = True) -> Array:
    """Record ``ak.argsort`` along ``axis``."""
    params: dict[str, ParamValue] = {"axis": axis, "ascending": ascending}
    if not stable:
        params["stable"] = False
    return arr.session.record_op("ak.argsort", [arr], params)


def local_index(arr: Array, axis: int = -1) -> Array:
    """Record ``ak.local_index`` along ``axis``."""
    return arr.session.record_op("ak.local_index", [arr], {"axis": axis})


def concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Record ``ak.concatenate`` of ``arrays`` along ``axis``."""
    return arrays[0].session.record_op("ak.concatenate", list(arrays), {"axis": axis})


def flatten(arr: Array, axis: int | None = 1) -> Array:
    """Record ``ak.flatten`` along ``axis`` (``None`` flattens fully)."""
    return arr.session.record_op("ak.flatten", [arr], {"axis": "none" if axis is None else axis})


def fill_none(arr: Array, value: object, axis: int = -1) -> Array:
    """Record ``ak.fill_none``, replacing missing values with ``value``."""
    params = {**scalar_params(value, "value", "value_dtype"), "axis": axis}
    return arr.session.record_op("ak.fill_none", [arr], params)


def drop_none(arr: Array, axis: int | None = None) -> Array:
    """Record ``ak.drop_none`` along ``axis``."""
    params: dict[str, ParamValue] = {} if axis is None else {"axis": axis}
    return arr.session.record_op("ak.drop_none", [arr], params)


def where(cond: Array, a: object, b: object) -> Array:
    """Record ``ak.where``; a scalar ``a`` or ``b`` travels as a param."""
    inputs: list[Array] = []
    params: dict[str, ParamValue] = {}
    for i, operand in enumerate((cond, a, b)):
        if isinstance(operand, Array):
            inputs.append(operand)
        else:
            params.update(scalar_params(operand, f"const{i}", f"const{i}_dtype"))
    return cond.session.record_op("ak.where", inputs, params)


def _dtype_name(dtype: npt.DTypeLike) -> str:
    """Return a dtype read as awkward reads one (``np.dtype``), recorded by name."""
    return np.dtype(dtype).name


def zeros_like(arr: Array, *, dtype: npt.DTypeLike = None) -> Array:
    """``ak.zeros_like`` parity: ``dtype=None`` PRESERVES the input dtype (no forced int64)."""
    params: dict[str, ParamValue] = {} if dtype is None else {"dtype": _dtype_name(dtype)}
    return arr.session.record_op("ak.zeros_like", [arr], params)


def ones_like(arr: Array, *, dtype: npt.DTypeLike = None) -> Array:
    """Record ``ak.ones_like``, optionally as ``dtype``."""
    params: dict[str, ParamValue] = {} if dtype is None else {"dtype": _dtype_name(dtype)}
    return arr.session.record_op("ak.ones_like", [arr], params)


# ---- M28: variadic call templates for the recorded externals ------------------------------------
# The template ("$i" slots, [..] groups, constants where the callee allows) is PRESERVED NODE
# CONTENT: it rides the IR as canonical JSON, and record-time evaluation materializes THE SAME
# template the preservation replay obeys — agreement by construction. Descriptors on this path use
# the content-identity hash conventions shared with graphed-preserve, and params carry no
# filesystem paths. The template-less calls below remain the M3 path, byte-for-byte.


@dataclass
class _TemplateExternal:
    """The template path's recorded evaluator — PICKLABLE, unlike the closure it replaced.

    ``call=None`` means "evaluate through the preserve plugin": the resource is rebuilt from
    ``payload`` and cached per process, which is exactly what a worker does after a stdlib pickle
    drops a live handle. Recording with ``call=None`` therefore makes in-process and out-of-process
    evaluation the same code by construction — a ``call`` that is NOT the plugin's own evaluation
    diverges silently between backends. ``apply_correction`` records ``None``; ``onnx_inference``
    still records the user's ``runner``, whose inference session may be configured (providers,
    devices) in ways the plugin's CPU-only ``load`` would not reproduce.

    No cloudpickle: these payloads are preservable, not opaque (§A.3.1).
    """

    kind: str
    payload: bytes
    params: dict[str, ParamValue]
    call: Callable[..., object] | None

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "call": None}

    def __call__(self, *values: object) -> object:
        if self.call is None:
            from ..preserve.externals import get_plugin
            from ..preserve.externals._base import _PluginEvaluator

            plugin = get_plugin(self.kind)
            if plugin is None:
                raise RuntimeError(
                    f"external payload kind {self.kind!r} has no registered plugin, so it "
                    "cannot be rebuilt from the payload bytes"
                )
            # _PluginEvaluator caches the loaded resource per process by payload and params, so the
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
    compound: bool = False,
) -> Array:
    """Record a correctionlib scale-factor application as an External node.

    ``compound=True`` names an entry of the set's ``compound_corrections`` rather than its
    ``corrections``; without it a regular correction wins and a compound-only name falls back to
    ``compound_corrections``.

    With ``args=`` (e.g. ``["nominal", "$0", "$1"]``): the M28 preservation-aligned path —
    content-identity descriptor, no path in the IR, inputs passed NATIVELY (awkward/numpy, jagged
    preserved) per the template, which replay obeys identically.
    Without it: the original M3 recording, unchanged (``payload`` must then be a path).

    ``evaluator`` is the eager evaluation for the ``args=None`` (M3) path ONLY. On the template
    path EVERY backend — in-process included — evaluates through the correctionlib preserve
    plugin, rebuilt from the payload bytes under the recorded template, so ``evaluator`` is
    ignored: the two cannot diverge, and in-process gets the plugin's flat-buffer evaluation
    instead of correctionlib's per-call ``ak.transform`` wrapper. The recorded type is the
    broadcast of every slotted input, as that wrapper broadcasts them.
    """
    # recorded only when set, so a regular correction's node identity is what it always was
    which: dict[str, ParamValue] = {"compound": True} if compound else {}
    if args is None:
        return inputs[0].session.record_external(
            "correction", evaluator, list(inputs), {"path": str(payload), "name": name, **which}
        )
    blob = _payload_bytes(payload)
    entries = _parse_template(args, len(inputs), constants=True, groups=False)
    slots = [inputs[i] for kind, i in entries if kind == "slot"]
    session = inputs[0].session
    session._mine(list(inputs), ("correction", capture()))  # before `session.form` reads an input
    descriptor = payloads.correctionlib_contents_descriptor(blob, name)
    params: dict[str, ParamValue] = {
        "name": name,
        "args": json.dumps(args, sort_keys=False),
        **which,
    }
    return session.record_external(
        "correction",
        # call=None: the plugin owns evaluation on every backend (see _TemplateExternal)
        _TemplateExternal(
            descriptor.kind, blob, {**params, "content_hash": descriptor.content_hash}, None
        ),
        list(inputs),
        params,
        descriptor=descriptor,
        # the correctionlib plugin's value is float64 (its outputs are `real`), checked at run time
        form=astype_form(
            broadcast_form([cast("AwkwardForm", session.form(a)) for a in slots]), "float64"
        ),
        form_params={"output_dtype": "float64"},
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
    in-process eager path, so it must be the plugin's own inference (a wrapper diverges by
    backend).
    """
    if args is None and kwargs is None:
        return inputs[0].session.record_external(
            "onnx", runner, list(inputs), {"path": str(payload)}
        )
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
    session._mine(list(inputs), ("onnx", capture()))  # before `session.form` reads an input
    params: dict[str, str] = {}
    if args is not None:
        params["args"] = json.dumps(args, sort_keys=False)
    if kwargs is not None:
        params["kwargs"] = json.dumps(kwargs, sort_keys=True)
    descriptor = payloads.onnx_weights_descriptor(blob)
    return session.record_external(
        "onnx",
        _TemplateExternal(
            descriptor.kind, blob, {**params, "content_hash": descriptor.content_hash}, _fn
        ),
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
    """Record ``ak.sort`` along ``axis``."""
    params: dict[str, ParamValue] = {"axis": axis, "ascending": ascending}
    if not stable:
        params["stable"] = False
    return arr.session.record_op("ak.sort", [arr], params)


def ravel(arr: Array) -> Array:
    """Record ``ak.ravel``."""
    return arr.session.record_op("ak.ravel", [arr])


def run_lengths(arr: Array) -> Array:
    """Record ``ak.run_lengths``."""
    return arr.session.record_op("ak.run_lengths", [arr])


def mask(arr: Array, condition: Array, *, valid_when: bool = True) -> Array:
    """Record ``ak.mask`` of ``arr`` by ``condition``."""
    return arr.session.record_op("ak.mask", [arr, condition], {"valid_when": valid_when})


def is_none(arr: Array, axis: int = 0) -> Array:
    """Record ``ak.is_none`` along ``axis``."""
    return arr.session.record_op("ak.is_none", [arr], {"axis": axis})


def singletons(arr: Array, axis: int = 0) -> Array:
    """Record ``ak.singletons`` along ``axis``."""
    return arr.session.record_op("ak.singletons", [arr], {"axis": axis})


def pad_none(arr: Array, target: int, axis: int = 1, *, clip: bool = False) -> Array:
    """Record ``ak.pad_none`` to length ``target`` along ``axis``."""
    return arr.session.record_op(
        "ak.pad_none", [arr], {"target": target, "axis": axis, "clip": clip}
    )


def unflatten(arr: Array, counts: Array | int, axis: int = 0) -> Array:
    """``ak.unflatten``'s counts is "an integer or a one-dimensional array of integers".

    An integer is a PARAMETER, not an input, so it never enters the graph as an edge.
    """
    if isinstance(counts, int):
        return arr.session.record_op("ak.unflatten", [arr], {"axis": axis, "counts": counts})
    return arr.session.record_op("ak.unflatten", [arr, counts], {"axis": axis})


def to_regular(arr: Array, axis: int = 1) -> Array:
    """Record ``ak.to_regular`` along ``axis``."""
    return arr.session.record_op("ak.to_regular", [arr], {"axis": axis})


def from_regular(arr: Array, axis: int = 1) -> Array:
    """Record ``ak.from_regular`` along ``axis``."""
    return arr.session.record_op("ak.from_regular", [arr], {"axis": axis})


def full_like(arr: Array, value: float, *, dtype: npt.DTypeLike = None) -> Array:
    """Record ``ak.full_like`` filled with ``value``, optionally as ``dtype``."""
    params = scalar_params(value, "value", "value_dtype")
    if dtype is not None:
        params["dtype"] = _dtype_name(dtype)
    return arr.session.record_op("ak.full_like", [arr], params)


def nan_to_num(
    arr: Array,
    copy: bool = True,  # ponytail: ignored, a recorded op never mutates its input
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> Array:
    """Record ``ak.nan_to_num``; ``copy`` is accepted and ignored."""
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
    """Record ``ak.isclose`` of ``x`` and ``y``."""
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
    """Record ``ak.argcombinations`` of ``n`` items per list."""
    return arr.session.record_op(
        "ak.argcombinations",
        [arr],
        _comb_params(n, replacement, axis, fields, with_name, parameters),
    )


def argcartesian(
    arrays: Sequence[Array] | Mapping[str, Array],
    *,
    axis: int = 1,
    nested: bool | Sequence[int] | Sequence[str] | None = None,
    with_name: str | None = None,
    parameters: Mapping[str, ParamValue] | None = None,
) -> Array:
    """Record ``ak.argcartesian`` over ``arrays``."""
    inputs = list(arrays.values() if isinstance(arrays, Mapping) else arrays)
    return inputs[0].session.record_op(
        "ak.argcartesian", inputs, _cartesian_params(arrays, axis, nested, with_name, parameters)
    )


def without_field(arr: Array, field: str) -> Array:
    """Record ``ak.without_field``, dropping ``field``."""
    return arr.session.record_op("ak.without_field", [arr], {"field": field})


def values_astype(arr: Array, dtype: npt.DTypeLike) -> Array:
    """Record ``ak.values_astype`` to ``dtype``."""
    return arr.session.record_op("ak.values_astype", [arr], {"dtype": _dtype_name(dtype)})


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
    out: list[object] = ak.to_list(arr.session.materialize(arr))
    return out


# ---- M18: behaviors (dask-awkward parity plan P2) ---------------------------------------------
def with_name(arr: Array, name: str) -> Array:
    """Name the records.

    With a behavior dict registered on the backend, behavior properties (vector's .pt/.mass) then
    work through plain attribute access.
    """
    return arr.session.record_op("ak.with_name", [arr], {"name": name})


def with_parameter(arr: Array, key: str, value: str | int | float | bool) -> Array:
    """Record ``ak.with_parameter`` setting ``key`` to ``value``."""
    return arr.session.record_op("ak.with_parameter", [arr], {"key": key, "value": value})


def without_parameters(arr: Array) -> Array:
    """Record ``ak.without_parameters``."""
    return arr.session.record_op("ak.without_parameters", [arr])


# ---- M19: introspection + peeking conveniences (parity plan P3.8) ----------------------------
def fields(arr: Array) -> list[str]:
    """Record-field names from the FORM (pure metadata: records nothing)."""
    return list(arr.session.form(arr).tt.fields)  # type: ignore[attr-defined]


def type_of(arr: Array) -> str:
    """Return the awkward type string from the FORM (pure metadata: records nothing)."""
    return str(arr.session.form(arr).describe())


def backend_of(arr: Array) -> str:
    """Return the session's backend class name (pure metadata: records nothing)."""
    return type(arr.session.backend).__name__


def head(arr: Array, n: int = 5) -> object:
    """EAGER peek at the first ``n`` rows (the common slice op + reference materialize)."""
    return arr.session.materialize(arr[:n])


def sample(arr: Array, *, factor: int) -> object:
    """EAGER peek at every ``factor``-th row."""
    return arr.session.materialize(arr[::factor])


# ---- §2.3c: the gak dispatch layer; `_dispositions` classifies each function ----------
def _dispatched(fn: Callable[..., Any], kind: str) -> Callable[..., Any]:
    """Wrap one gak function with the dispatch its classification names (§2.3c).

    Signatures do not change (R17.0's anti-drift pin): `functools.wraps` keeps `__wrapped__`, so
    `inspect.signature` still reports the underlying one.
    """

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
