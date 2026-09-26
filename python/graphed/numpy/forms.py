"""The numpy backend's form: dtype + shape with a partitioned (unknown-length) axis 0 (M11)."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from graphed.backend import PYTHON_TYPES


@dataclass(frozen=True)
class NumpyForm:
    """Form for a numpy array: dtype + shape (axis 0 is the partitioned axis, length ``None``),
    or a record source (named columns)."""

    dtype: np.dtype
    kind: str = "vector"
    fields: tuple[tuple[str, str], ...] | None = None  # (column, dtype-str) for record sources
    shape: tuple[int | None, ...] = (None,)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def describe(self) -> str:
        if self.fields is not None:
            return f"record[{','.join(f for f, _ in self.fields)}]"
        if len(self.shape) > 1:
            return f"{self.kind}[{self.dtype}, shape={self.shape}]"
        return f"{self.kind}[{self.dtype}]"  # the M2 pin: 1-D stays vector[<dtype>]


def is_numeric(form: NumpyForm) -> bool:
    return form.fields is None and np.issubdtype(form.dtype, np.number)


def meta(form: NumpyForm) -> Any:
    """A zero-length stand-in carrying the form's dtype/shape: numpy itself infers the result."""
    if form.fields is not None:
        raise TypeError(f"elementwise ops need array operands, got {form.describe()}; access a field first")
    return np.empty(tuple(0 if d is None else d for d in form.shape), dtype=form.dtype)


def unit_meta(form: NumpyForm) -> Any:
    """A LENGTH-ONE stand-in for reduction inference (argmin/mean reject zero-length input)."""
    if form.fields is not None:
        raise TypeError(f"reductions need array operands, got {form.describe()}; access a field first")
    return np.zeros(tuple(1 if d is None else d for d in form.shape), dtype=form.dtype)


def form_from_meta(result: object, leading_none: bool) -> NumpyForm:
    arr = np.asarray(result)
    if arr.ndim == 0:
        return NumpyForm(arr.dtype, kind="scalar", shape=())
    shape: tuple[int | None, ...] = ((None,) if leading_none else (arr.shape[0],)) + arr.shape[1:]
    return NumpyForm(arr.dtype, shape=shape)


def _dtype(spec: Any) -> np.dtype:
    """`np.dtype`, also of its own `str` for structured/subarray dtypes (a Python literal)."""
    try:
        d: np.dtype = np.dtype(spec)
    except (TypeError, ValueError):
        if not isinstance(spec, str):
            raise
        d = np.dtype(ast.literal_eval(spec))
    return d


def canonical_output_type(spec: object) -> str:
    """Reduce an ``output_type`` spelling (numpy dtype-like, Python type, or an awkward type object
    that is a dtype) to ``str`` of its numpy dtype, a fixpoint under `_dtype`."""
    s: Any = spec
    if isinstance(s, type) and s in PYTHON_TYPES:
        return PYTHON_TYPES[s]
    if type(s).__module__.startswith("awkward."):  # the object exists, so awkward is loaded
        from graphed.awkward.backend import canonical_output_type as awkward_canonical  # noqa: PLC0415

        s = awkward_canonical(s)
    try:
        return str(_dtype(s))
    except (TypeError, ValueError, SyntaxError) as exc:
        raise TypeError(
            f"NumpyBackend cannot represent output_type {s!r}; it takes a numpy dtype or Python scalar type"
        ) from exc


def declared_form(first: NumpyForm, canonical: str) -> NumpyForm:
    """The form of a `map` declared ``canonical``: that element over ``first``'s leading axis."""
    d = _dtype(canonical)
    if first.kind == "scalar" and (d.names is not None or d.subdtype is not None):
        raise TypeError(f"output_type {canonical!r} over a scalar input must be a primitive dtype")
    if d.names is not None:
        if any(d[n].names is not None or d[n].subdtype is not None for n in d.names):
            raise TypeError(f"output_type {canonical!r}: numpy record columns are plain dtypes")
        return NumpyForm(np.dtype(object), kind="record", fields=tuple((n, d[n].str) for n in d.names))
    kind = "scalar" if first.kind == "scalar" else "vector"  # a record input's kind does not carry over
    return NumpyForm(d.base, kind=kind, shape=first.shape[:1] + d.shape)


def check_output_type(value: object, inputs: Sequence[object], key: str, declared: str) -> str | None:
    """``None`` when a `map`'s ``value`` has its declared type, else the value's description.

    The dtype must be equal, an unsized string (``str``, ``<U0``) matching any length of its kind;
    a subarray declaration pins the trailing shape. The leading axis is the first input's: none over
    a scalar. A record declaration takes a structured array or a column mapping with those fields.
    This backend records no ``output_dtype``, so it checks none."""
    if key != "output_type":
        return None
    d = _dtype(declared)
    first = inputs[0] if inputs else None
    lead = 0 if np.ndim(first) == 0 and not isinstance(first, Mapping) else 1  # a record source is a dict
    if isinstance(value, Mapping):
        cols = tuple((str(k), np.asarray(v).dtype) for k, v in value.items())
        ok = d.names is not None and cols == tuple((n, d[n]) for n in d.names)
        return None if ok else f"record[{','.join(f'{k}: {t}' for k, t in cols)}]"
    arr = np.asarray(value)
    if d.names is not None:  # field by field: a packed and a padded layout are one record
        same = arr.dtype.names == d.names and all(arr.dtype[n] == d[n] for n in d.names)
    else:
        want = d.base
        same = arr.dtype == want or (want.itemsize == 0 and want.kind in "SU" and arr.dtype.kind == want.kind)
    shape = arr.ndim == lead + len(d.shape) and arr.shape[lead:] == d.shape
    return None if same and shape else form_from_meta(arr, lead == 1).describe()
