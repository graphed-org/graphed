"""M11: record-time form inference via zero-length meta arrays (dask.array parity P0.1).

``NumpyForm`` carries a real ``(shape, dtype)`` — the partitioned axis 0 is ``None`` — and
``op_form`` infers result forms by evaluating the op on zero-length **meta arrays**, so dtype
promotion, broadcasting, and domain/type errors are numpy's own (plan §A.2: reuse the host
library's inference; never hand-roll it). Ill-typed programs fail AT RECORD TIME with a
``GraphedTypeError`` pointing at the user's line, before any data is touched.
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import GraphedTypeError, Session

from graphed_numpy import NumpyBackend, from_array


def test_dtype_promotion_is_numpys_own() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1, 2], dtype=np.int32))
    b = from_array(s, "b", np.array([1.0, 2.0], dtype=np.float64))
    out = a + b
    assert out.dtype == np.dtype(np.float64)
    assert np.exp(a).dtype == np.dtype(np.float64)  # int -> float, numpy's rule


def test_comparison_forms_are_boolean() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    assert (a > 1.0).dtype == np.dtype(np.bool_)
    assert np.isnan(a).dtype == np.dtype(np.bool_)


def test_shape_dtype_ndim_properties_on_sources_and_ops() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.ones((4, 3)))
    assert a.shape == (None, 3)  # axis 0 is the partitioned (unknown-length) axis
    assert a.ndim == 2
    assert a.dtype == np.dtype(np.float64)
    out = np.sqrt(a)
    assert out.shape == (None, 3)
    n = s.node_count()
    _ = out.shape, out.dtype, out.ndim
    assert s.node_count() == n  # metadata answers record nothing


def test_broadcasting_shapes_are_numpys_own() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.ones((4, 3)))
    b = from_array(s, "b", np.ones((4, 1)))
    assert (a + b).shape == (None, 3)
    assert (a + 2.0).shape == (None, 3)


def test_ill_typed_op_fails_at_record_time() -> None:
    s = Session(NumpyBackend())
    text = from_array(s, "t", np.array(["x", "y"]))
    with pytest.raises(GraphedTypeError):
        np.exp(text)


def test_unbroadcastable_shapes_fail_at_record_time() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.ones((4, 3)))
    b = from_array(s, "b", np.ones((4, 4)))
    with pytest.raises(GraphedTypeError):
        _ = a + b


def test_reduction_form_is_scalar() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    total = a.reduce("sum")
    assert s.form(total).describe().startswith("scalar[")
    assert total.shape == ()


def test_flat_vector_describe_is_unchanged() -> None:
    # the M2 pin: 1-D sources keep the exact vector[<dtype>] describe form
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    assert s.form(a).describe() == "vector[float64]"
