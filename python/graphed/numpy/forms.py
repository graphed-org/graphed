"""The numpy backend's form: dtype + shape with a partitioned (unknown-length) axis 0 (M11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
