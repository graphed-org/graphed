"""How gak keyword arguments become recorded op params, and the reduction recorders.

Shared by the ``gak`` functions in :mod:`graphed.awkward.functions`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from graphed import Array, ParamValue
from graphed.array import scalar_params


def crosses_rows(arr: Array, axis: int | None) -> bool:
    """Return whether ``axis`` of ``arr`` is the row axis, so a chunk's value is not the whole's.

    ``None`` is every axis, the row axis among them. A negative axis counts up from each field's
    own depth, so it is the row axis wherever a field is that deep.
    """
    if axis is None:
        return True
    if axis >= 0:
        return axis == 0
    tracer = arr.session.form(arr).tt  # type: ignore[attr-defined]
    low, high = tracer.layout.minmax_depth if hasattr(tracer, "layout") else (0, 0)
    return bool(low <= -axis <= high)


def _along_axis(
    name: str, inputs: Sequence[Array], params: Mapping[str, ParamValue], axis: int | None
) -> Array:
    """Record ``name`` along ``axis`` of ``inputs``, a reduction when that is the row axis."""
    head = inputs[0]
    return head.session.record_op(name, list(inputs), params, reduction=crosses_rows(head, axis))


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


def _cartesian_params(
    arrays: Sequence[Array] | Mapping[str, Array],
    axis: int,
    nested: bool | Sequence[int] | Sequence[str] | None,
    with_name: str | None,
    parameters: Mapping[str, ParamValue] | None,
) -> dict[str, ParamValue]:
    params: dict[str, ParamValue] = {}
    keys = list(arrays) if isinstance(arrays, Mapping) else []
    if keys:
        params["fields"] = ",".join(keys)
    if isinstance(nested, bool):
        params["nested"] = nested
    elif nested is not None:
        # a dict's slots nest by key, as ak.cartesian's do; the record keeps their positions
        params["nested_axes"] = ",".join(
            str(keys.index(str(i)) if keys else int(i)) for i in nested
        )
    if axis != 1:
        params["axis"] = axis
    params.update(_structure_params(with_name=with_name, parameters=parameters))
    return params


def _structure_params(
    *, with_name: str | None, parameters: Mapping[str, ParamValue] | None
) -> dict[str, ParamValue]:
    """Return the shared inline record-naming/parameter kwargs.

    Present-only; JSON for the dict.
    """
    out: dict[str, ParamValue] = {}
    if with_name is not None:
        out["with_name"] = with_name
    if parameters:
        out["parameters"] = json.dumps(dict(parameters), sort_keys=True)
    return out


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
    """Record one awkward reduction under the M12/M16 STRUCTURAL RULE.

    Reducing over the event (partitioned) axis — axis None or 0 — is a stage boundary executed by
    the M7 tree reduction; an inner-axis (per-event) reduction is partition-local and fusible. The
    ak parity kwargs are recorded PRESENT-ONLY (awkward's own defaults apply when absent); a weight
    is a second INPUT, not a parameter.
    """
    params: dict[str, object] = {"axis": "none" if axis is None else axis}
    if extra:
        params.update(extra)
    if keepdims:
        params["keepdims"] = True
    if mask_identity is not None:
        params["mask_identity"] = mask_identity
    if initial is not None:
        params.update(scalar_params(initial, "initial", "initial_dtype"))
    inputs = [arr] if weight is None else [arr, weight]
    if weight is not None:
        params["weighted"] = True
    return _along_axis(name, inputs, params, axis)  # type: ignore[arg-type]


def _reduce2(name: str, x: Array, y: Array, axis: int | None) -> Array:
    return _along_axis(name, [x, y], {"axis": "none" if axis is None else axis}, axis)
