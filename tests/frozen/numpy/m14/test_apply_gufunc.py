"""M14: ``apply_gufunc`` — the signature-aware numpy escape hatch (dask.array parity P3.8).

A gufunc signature ("(i),(i)->()") gives an OPAQUE callable a record-time form: input core dims
are bound against the operand forms (mismatches fail at record time, before any data), the output
core dims must be bound by inputs, and the caller declares the output dtype. The node is an
External whose PayloadDescriptor carries the signature as its io_schema — the opaque callable
stays a flagged preservation risk. The callable receives whole (N, ...) arrays (dask's
``vectorize=False`` semantics).
"""

from __future__ import annotations

import warnings

import graphed_core
import numpy as np
import pytest
from graphed import GraphedTypeError, ProjectionError, Session

import graphed_numpy as gn
from graphed_numpy import NumpyBackend, from_array

P3 = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0], [3.0, 5.0, 8.0]])
Q3 = np.array([[2.0, 7.0, 1.0], [8.0, 2.0, 8.0], [1.0, 8.0, 2.0], [8.0, 4.0, 5.0]])


def _s() -> Session:
    return Session(NumpyBackend())


def _rowdot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * y, axis=-1)


def test_pair_reduce_signature() -> None:
    s = _s()
    a = from_array(s, "a", P3)
    b = from_array(s, "b", Q3)
    out = gn.apply_gufunc(_rowdot, "(i),(i)->()", a, b, output_dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), _rowdot(P3, Q3))
    assert out.shape == (None,)
    assert out.dtype == np.dtype(np.float64)


def test_elementwise_and_preserving_signatures() -> None:
    s = _s()
    a = from_array(s, "a", P3[:, 0])
    soft = gn.apply_gufunc(np.exp, "()->()", a, output_dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(s.materialize(soft)), np.exp(P3[:, 0]))
    assert soft.shape == (None,)

    m = from_array(s, "m", P3)

    def norm_rows(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    out = gn.apply_gufunc(norm_rows, "(i)->(i)", m, output_dtype=np.float64)
    np.testing.assert_allclose(np.asarray(s.materialize(out)), norm_rows(P3))
    assert out.shape == (None, 3)


def test_matrix_reduce_signature_binds_output_dims() -> None:
    s = _s()
    stackd = np.stack([P3[:2], Q3[:2]], axis=0)  # (2, 2, 3): two events of (2, 3) matrices
    m = from_array(s, "m", stackd)

    def colsum(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-2)

    out = gn.apply_gufunc(colsum, "(i,j)->(j)", m, output_dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), colsum(stackd))
    assert out.shape == (None, 3)


def test_descriptor_carries_the_signature_as_io_schema() -> None:
    s = _s()
    a = from_array(s, "a", P3)
    b = from_array(s, "b", Q3)
    out = gn.apply_gufunc(_rowdot, "(i),(i)->()", a, b, output_dtype=np.float64, name="rowdot")
    g = graphed_core.GraphStore.deserialize(s.serialized_ir(out, optimize=False))
    (node,) = [n for n in g.nodes() if n["id"] == out.node_id]
    assert node["kind"] == "external"
    assert node["descriptor"]["io_schema"] == "(i),(i)->()"
    assert node["descriptor"]["kind"] == "opaque_callable"
    assert "rowdot" in node["descriptor"]["content_hash"]


def test_dim_binding_mismatches_fail_at_record_time() -> None:
    s = _s()
    a = from_array(s, "a", P3)  # inner dim 3
    b = from_array(s, "b", np.ones((4, 4)))  # inner dim 4
    with pytest.raises(GraphedTypeError):
        gn.apply_gufunc(_rowdot, "(i),(i)->()", a, b, output_dtype=np.float64)
    with pytest.raises(GraphedTypeError):
        gn.apply_gufunc(_rowdot, "(i)->(j)", a, output_dtype=np.float64)  # j unbound
    with pytest.raises(GraphedTypeError):
        gn.apply_gufunc(_rowdot, "(i,j)->(j)", a, output_dtype=np.float64)  # rank mismatch
    with pytest.raises(GraphedTypeError):
        gn.apply_gufunc(_rowdot, "(i),(i)->()", a, output_dtype=np.float64)  # arity mismatch


def test_malformed_signatures_are_refused() -> None:
    s = _s()
    a = from_array(s, "a", P3)
    with pytest.raises((TypeError, GraphedTypeError)):
        gn.apply_gufunc(_rowdot, "(i)", a, output_dtype=np.float64)  # no ->
    with pytest.raises((TypeError, GraphedTypeError)):
        gn.apply_gufunc(_rowdot, "(i)->(),()", a, output_dtype=np.float64)  # multi-output


def test_projection_stays_honest_through_gufunc_nodes() -> None:
    # an opaque gufunc node cannot be seen through: the M5 on-fail policy applies unchanged
    # (raise -> ProjectionError; warn -> CONSERVATIVE all columns; pass -> optimistic union)
    s = _s()
    events = gn.from_record(s, "events", pt=np.array([1.0, 2.0]), eta=np.array([0.1, 0.2]))
    out = gn.apply_gufunc(np.exp, "()->()", events["pt"], output_dtype=np.float64)
    with pytest.raises(ProjectionError):
        gn.project(out, on_fail="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conservative = gn.project(out, on_fail="warn")
    assert conservative.columns_for("events") == frozenset({"pt", "eta"})  # everything
    optimistic = gn.project(out, on_fail="pass")
    assert optimistic.columns_for("events") == frozenset({"pt"})  # the union through the node
