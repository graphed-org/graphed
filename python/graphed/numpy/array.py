"""``NumpyArray``: the numpy-idiomatic deferred-array proxy (M11, dask.array parity P0).

The shared ``graphed.Array`` is backend-idiom-neutral — it carries operators, ufunc dispatch, and
protected infrastructure, but no numpy-specific surface (awkward's idiom applies operations as
functions over arrays, never as member functions). This subclass COMPLETES the numpy idiom for the
numpy backend: metadata properties (``.shape/.dtype/.ndim`` answered from the form) and the
``__array_function__`` protocol so numpy API calls (``np.sum(a)``, …) record canonical ops.
``NumpyBackend.array_type`` hands this class to the Session, so every recorded node comes back as
a ``NumpyArray``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graphed import Array

Handler = Callable[["NumpyArray", tuple[object, ...], dict[str, object]], Array]


class NumpyArray(Array):
    """A deferred array with numpy's calling idiom (methods, properties, numpy API dispatch)."""

    __slots__ = ()

    # ---- array metadata (answered from the form; never recorded) ----------------
    @property
    def shape(self) -> Any:
        return self._form_meta("shape")

    @property
    def dtype(self) -> Any:
        return self._form_meta("dtype")

    @property
    def ndim(self) -> Any:
        return self._form_meta("ndim")

    # ---- numpy API dispatch ------------------------------------------------------
    def __array_function__(
        self, func: Any, types: object, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> Any:
        handler = _NUMPY_FUNCTIONS.get(getattr(func, "__name__", ""))
        if handler is None:
            return NotImplemented  # numpy raises TypeError naming the unsupported function
        return handler(self, args, kwargs)


def _fn_sum(arr: NumpyArray, args: tuple[object, ...], kwargs: dict[str, object]) -> Array:
    if len(args) != 1 or args[0] is not arr or kwargs:
        raise TypeError("graphed records np.sum(array) with no extra arguments (axis-aware reductions: M12)")
    return arr.reduce("sum")


# numpy API function name -> recorder. M11 wires the protocol + whole-array sum; M12 extends with
# the axis-aware reduction/scan tier, M13 with manipulation routines, without re-touching it.
_NUMPY_FUNCTIONS: dict[str, Handler] = {
    "sum": _fn_sum,
}
