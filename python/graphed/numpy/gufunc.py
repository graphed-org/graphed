"""``apply_gufunc``: the signature-aware opaque escape hatch (M14, dask.array parity P3.8).

A gufunc signature ("(i),(i)->()") is what makes an OPAQUE callable typable at record time: each
input's core dims are bound against the operand forms' inner dims (the partitioned axis 0 is the
implicit loop dim), the single output's core dims must be bound by inputs, and the caller declares
the output dtype. The recorded node is an External whose ``PayloadDescriptor`` carries the
signature as its ``io_schema`` — the callable stays a flagged preservation risk (plan A.3.1).

Evaluation hands the callable whole ``(N, ...)`` arrays (dask's ``vectorize=False`` semantics):
the function must already be vectorized over the loop dim.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from graphed import Array

from .array import NumpyArray
from .forms import NumpyForm

_GROUP = re.compile(r"\(([^)]*)\)")

CoreDims = tuple[str, ...]


def parse_signature(signature: str) -> tuple[list[CoreDims], CoreDims]:
    lhs, arrow, rhs = signature.partition("->")
    if not arrow:
        raise TypeError(f"gufunc signature needs '->': {signature!r}")

    def side(text: str) -> list[CoreDims]:
        groups = _GROUP.findall(text)
        if not groups or _GROUP.sub("", text).strip(", "):
            raise TypeError(f"malformed gufunc signature: {signature!r}")
        return [tuple(d.strip() for d in grp.split(",") if d.strip()) for grp in groups]

    ins, outs = side(lhs), side(rhs)
    if len(outs) != 1:
        raise TypeError("the MVP gufunc supports exactly ONE output")
    return ins, outs[0]


def gufunc_form(forms: Sequence[NumpyForm], params: Any) -> NumpyForm:
    """Record-time form inference for a gufunc node (used by ``NumpyBackend.op_form``)."""
    ins, out = parse_signature(str(params["signature"]))
    if len(ins) != len(forms):
        raise TypeError(f"signature has {len(ins)} inputs but {len(forms)} operands were given")
    bound: dict[str, int | None] = {}
    for form, core in zip(forms, ins, strict=True):
        if form.fields is not None:
            raise TypeError(f"gufunc operands must be arrays, got {form.describe()}")
        inner = form.shape[1:]
        if len(inner) != len(core):
            raise TypeError(
                f"operand of shape {form.shape} has {len(inner)} core dim(s); signature wants {len(core)}"
            )
        for name, size in zip(core, inner, strict=True):
            if name in bound and bound[name] != size:
                raise TypeError(f"core dim {name!r} bound to both {bound[name]} and {size}")
            bound.setdefault(name, size)
    out_dims: list[int | None] = []
    for name in out:
        if name not in bound:
            raise TypeError(f"output core dim {name!r} is not bound by any input")
        out_dims.append(bound[name])
    return NumpyForm(np.dtype(str(params["dtype"])), shape=(None, *out_dims))


def apply_gufunc(
    fn: Callable[..., object],
    signature: str,
    *arrays: NumpyArray,
    output_dtype: object,
    name: str | None = None,
) -> Array:
    """Record ``fn`` over ``arrays`` with gufunc-signature form inference (one External node)."""
    parse_signature(signature)  # malformed signatures fail HERE, before any recording
    if not arrays or not all(isinstance(a, NumpyArray) for a in arrays):
        raise TypeError("apply_gufunc needs at least one deferred array operand")
    session = arrays[0].session
    if any(a.session is not session for a in arrays):
        raise TypeError("apply_gufunc operands must come from one Session")
    fn_name: str = name or str(getattr(fn, "__name__", "lambda"))
    params = {"fn": fn_name, "signature": signature, "dtype": np.dtype(output_dtype).str}
    return session.record_external("gufunc", fn, list(arrays), params)
