"""M11: the numpy calling idiom lives on ``graphed_numpy.NumpyArray``, not on ``graphed.Array``.

The factorization pin (design review 2026-06-10): the shared frontend proxy is backend-idiom-
neutral; ``NumpyBackend.array_type`` supplies the subclass that completes numpy's method/property
surface, and every Session builder returns it.
"""

from __future__ import annotations

import numpy as np
from graphed import Array, Session

from graphed_numpy import NumpyArray, NumpyBackend, from_array, from_record


def test_every_builder_returns_the_numpy_proxy() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    r = from_record(s, "events", pt=np.array([1.0]))
    assert type(a) is NumpyArray
    assert type(r) is NumpyArray
    assert type(a + a) is NumpyArray  # record_op
    assert type(np.exp(a)) is NumpyArray  # ufunc dispatch
    assert type(a.map(lambda x: x)) is NumpyArray  # record_external
    assert type(r["pt"]) is NumpyArray  # field access


def test_the_idiom_is_not_on_the_shared_proxy() -> None:
    for name in ("shape", "dtype", "ndim", "__array_function__"):
        assert name in vars(NumpyArray)
        assert name not in vars(Array), f"numpy-idiomatic {name!r} leaked onto graphed.Array"
