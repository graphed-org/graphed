"""Creation routines (M12, dask.array parity P1.5).

Concrete creators (`zeros/ones/full/empty/arange/linspace`) record **deterministically named
sources** — the name encodes the content, so identical creations intern to the SAME source node
and creation programs serialize byte-identically. `*_like` creators record **fusible ops** (the
operand's length along the partitioned axis 0 is unknown at record time). `empty`/`empty_like`
are deterministic: they are zeros — uninitialized memory would break the byte-identity gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from graphed import Array, Session

from .forms import NumpyForm

ShapeLike = int | tuple[int, ...]


def _norm_shape(shape: ShapeLike) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else tuple(int(d) for d in shape)


def _source(session: Session, name: str, arr: Any) -> Array:
    return session.source(name, form=NumpyForm(arr.dtype, shape=(None, *arr.shape[1:])), data=arr)


def zeros(session: Session, shape: ShapeLike, dtype: Any = float, *, name: str | None = None) -> Array:
    arr = np.zeros(_norm_shape(shape), dtype)
    return _source(session, name or f"zeros[{arr.shape},{arr.dtype}]", arr)


def ones(session: Session, shape: ShapeLike, dtype: Any = float, *, name: str | None = None) -> Array:
    arr = np.ones(_norm_shape(shape), dtype)
    return _source(session, name or f"ones[{arr.shape},{arr.dtype}]", arr)


def full(
    session: Session, shape: ShapeLike, fill_value: float, dtype: Any = None, *, name: str | None = None
) -> Array:
    arr = np.full(_norm_shape(shape), fill_value, dtype)
    return _source(session, name or f"full[{arr.shape},{arr.dtype},{fill_value!r}]", arr)


def empty(session: Session, shape: ShapeLike, dtype: Any = float, *, name: str | None = None) -> Array:
    """Deterministic ``empty``: ZEROS by design (the determinism gate forbids uninitialized data)."""
    arr = np.zeros(_norm_shape(shape), dtype)
    return _source(session, name or f"empty[{arr.shape},{arr.dtype}]", arr)


_np_arange: Any = np.arange  # numpy's overload set fights float-typed inputs under --strict


def arange(
    session: Session,
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: Any = None,
    *,
    name: str | None = None,
) -> Array:
    arr = _np_arange(start, stop, step, dtype) if stop is not None else _np_arange(start, dtype=dtype)
    return _source(session, name or f"arange[{start},{stop},{step},{arr.dtype}]", arr)


def linspace(
    session: Session, start: float, stop: float, num: int = 50, dtype: Any = None, *, name: str | None = None
) -> Array:
    arr = np.linspace(start, stop, num, dtype=dtype)
    return _source(session, name or f"linspace[{start},{stop},{num},{arr.dtype}]", arr)


def zeros_like(array: Array) -> Array:
    return array.session.record_op("zeros_like", [array])


def ones_like(array: Array) -> Array:
    return array.session.record_op("ones_like", [array])


def empty_like(array: Array) -> Array:
    """Deterministic ``empty_like``: recorded as ``zeros_like`` (same rationale as ``empty``)."""
    return array.session.record_op("zeros_like", [array])


def full_like(array: Array, fill_value: float) -> Array:
    return array.session.record_op("full_like", [array], {"fill": float(fill_value)})
