"""Creation routines (M12, dask.array parity P1.5).

Concrete creators (`zeros/ones/full/empty/arange/linspace`) record **deterministically named
sources** — the name encodes the content, so identical creations intern to the SAME source node
and creation programs serialize byte-identically. `*_like` creators record **fusible ops** (the
operand's length along the partitioned axis 0 is unknown at record time). `empty`/`empty_like`
are deterministic: they are zeros — uninitialized memory would break the byte-identity gate.
"""

from __future__ import annotations

from typing import Any, cast, overload

import numpy as np

from graphed import Array, Session, broadcasting
from graphed.array import _scalar_params
from graphed.session import _reader

from . import NumpyBackend
from .forms import NumpyForm

ShapeLike = int | tuple[int, ...]


def _norm_shape(shape: ShapeLike) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else tuple(int(d) for d in shape)


def _exact(value: Any) -> str:
    """Render ``value`` for a content name so that only equal values render alike.

    ``repr`` of an ndarray rounds to 8 digits and elides past 1000 elements; ``tolist`` does
    neither.
    """
    a = np.asarray(value)
    return repr(a.item()) if a.ndim == 0 else f"{a.dtype}{a.shape}:{a.tolist()!r}"


def _source(session: Session, name: str | None, content: str, arr: Any) -> Array:
    # the content name fixes the values, so an identical creation reuses the held binding
    form = NumpyForm(arr.dtype, shape=(None, *arr.shape[1:]))
    return session.source(name or content, form=form, data=arr, identity=content)


@overload
def zeros(
    session: Session, shape: ShapeLike, dtype: Any = float, *, name: str | None = None
) -> Array: ...
@overload
def zeros(
    shape: ShapeLike,
    dtype: Any = float,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array: ...
@_reader(NumpyBackend)
def zeros(
    shape: ShapeLike,
    dtype: Any = float,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array:
    """Create a source array of zeros."""
    arr = np.zeros(_norm_shape(shape), dtype)
    return _source(cast("Session", session), name, f"zeros[{arr.shape},{arr.dtype}]", arr)


@overload
def ones(
    session: Session, shape: ShapeLike, dtype: Any = float, *, name: str | None = None
) -> Array: ...
@overload
def ones(
    shape: ShapeLike,
    dtype: Any = float,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array: ...
@_reader(NumpyBackend)
def ones(
    shape: ShapeLike,
    dtype: Any = float,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array:
    """Create a source array of ones."""
    arr = np.ones(_norm_shape(shape), dtype)
    return _source(cast("Session", session), name, f"ones[{arr.shape},{arr.dtype}]", arr)


@overload
def full(
    session: Session,
    shape: ShapeLike,
    fill_value: Any,
    dtype: Any = None,
    *,
    name: str | None = None,
) -> Array: ...
@overload
def full(
    shape: ShapeLike,
    fill_value: Any,
    dtype: Any = None,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array: ...
@_reader(NumpyBackend)
def full(
    shape: ShapeLike,
    fill_value: Any,
    dtype: Any = None,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array:
    """Create a source array filled with ``fill_value``."""
    arr = np.full(_norm_shape(shape), fill_value, dtype)
    return _source(
        cast("Session", session), name, f"full[{arr.shape},{arr.dtype},{_exact(fill_value)}]", arr
    )


@overload
def empty(
    session: Session, shape: ShapeLike, dtype: Any = float, *, name: str | None = None
) -> Array: ...
@overload
def empty(
    shape: ShapeLike,
    dtype: Any = float,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array: ...
@_reader(NumpyBackend)
def empty(
    shape: ShapeLike,
    dtype: Any = float,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array:
    """Deterministic ``empty``: ZEROS by design.

    The determinism gate forbids uninitialized data.
    """
    arr = np.zeros(_norm_shape(shape), dtype)
    return _source(cast("Session", session), name, f"empty[{arr.shape},{arr.dtype}]", arr)


_np_arange: Any = np.arange  # numpy's overload set fights float-typed inputs under --strict


@overload
def arange(
    session: Session,
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: Any = None,
    *,
    name: str | None = None,
) -> Array: ...
@overload
def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: Any = None,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array: ...
@_reader(NumpyBackend)
def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: Any = None,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array:
    """Create a source array of evenly spaced values, as ``numpy.arange``."""
    arr = (
        _np_arange(start, stop, step, dtype)
        if stop is not None
        else _np_arange(start, dtype=dtype)
    )
    return _source(
        cast("Session", session),
        name,
        f"arange[{_exact(start)},{_exact(stop)},{_exact(step)},{arr.dtype}]",
        arr,
    )


@overload
def linspace(
    session: Session,
    start: float,
    stop: float,
    num: int = 50,
    dtype: Any = None,
    *,
    name: str | None = None,
) -> Array: ...
@overload
def linspace(
    start: float,
    stop: float,
    num: int = 50,
    dtype: Any = None,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array: ...
@_reader(NumpyBackend)
def linspace(
    start: float,
    stop: float,
    num: int = 50,
    dtype: Any = None,
    *,
    name: str | None = None,
    session: Session | None = None,
) -> Array:
    """Create a source array of ``num`` evenly spaced values, as ``numpy.linspace``."""
    arr = np.linspace(start, stop, num, dtype=dtype)
    return _source(
        cast("Session", session),
        name,
        f"linspace[{_exact(start)},{_exact(stop)},{_exact(num)},{arr.dtype}]",
        arr,
    )


@broadcasting
def zeros_like(array: Array) -> Array:
    """Record zeros shaped like ``array``."""
    return array.session.record_op("zeros_like", [array])


@broadcasting
def ones_like(array: Array) -> Array:
    """Record ones shaped like ``array``."""
    return array.session.record_op("ones_like", [array])


@broadcasting
def empty_like(array: Array) -> Array:
    """Deterministic ``empty_like``: recorded as ``zeros_like`` (same rationale as ``empty``)."""
    return array.session.record_op("zeros_like", [array])


@broadcasting
def full_like(array: Array, fill_value: object) -> Array:
    """Record ``fill_value`` shaped like ``array``."""
    return array.session.record_op(
        "full_like", [array], _scalar_params(fill_value, "fill", "fill_dtype")
    )
