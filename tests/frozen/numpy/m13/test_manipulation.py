"""M13: in-partition manipulation parity (dask.array parity P2.6).

Every op evaluates exactly as numpy. Geometry rules of the axis-0-partitioned MVP are pinned as
record-time refusals: the partitioned axis cannot be moved (transpose/swapaxes/expand_dims at
axis 0), reshape must keep it leading (first target dim -1), squeeze needs an explicit inner
axis, and stack/vstack (which would create an inner unknown-length dim) are refused outright —
Phase 2's N-D chunking lifts these.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphed import GraphedTypeError, Session
from graphed.numpy import NumpyBackend, from_array

V = np.array([5.0, 1.0, 4.0, 2.0, 8.0, 6.0, 3.0, 7.0, 9.0, 0.0, 2.5, 1.5])
W = np.array([2.0, 0.5, 3.5, 1.0, 7.5, 6.5, 2.5, 8.0, 9.5, 0.5, 3.0, 1.0])
D2 = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0], [3.0, 5.0, 8.0]])
D3 = np.arange(24.0).reshape(2, 3, 4)


def _s() -> Session:
    return Session(NumpyBackend())


def test_reshape_keeps_the_partitioned_axis_leading() -> None:
    s = _s()
    a = from_array(s, "a", V)
    out = a.reshape(-1, 3)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), V.reshape(-1, 3))
    assert out.shape == (None, 3)
    assert np.reshape(a, (-1, 3)).node_id == out.node_id  # function and method intern
    with pytest.raises(GraphedTypeError):
        a.reshape(4, 3)  # a concrete leading dim cannot be checked against an unknown length
    with pytest.raises(GraphedTypeError):
        a.reshape(3, -1)


def test_ravel_flattens_to_the_partitioned_axis() -> None:
    s = _s()
    d3 = from_array(s, "d3", D3)
    out = d3.ravel()
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), D3.ravel())
    assert out.shape == (None,)
    assert np.ravel(d3).node_id == out.node_id


def test_squeeze_requires_an_explicit_inner_axis() -> None:
    s = _s()
    a = from_array(s, "a", V.reshape(12, 1))
    out = a.squeeze(axis=1)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), V)
    assert out.shape == (None,)
    assert np.squeeze(a, axis=1).node_id == out.node_id
    with pytest.raises(TypeError):
        a.squeeze()  # squeezing ALL size-1 dims could silently eat a length-1 partition
    with pytest.raises(GraphedTypeError):
        a.squeeze(axis=0)


def test_expand_dims_inner_axes_only() -> None:
    s = _s()
    d2 = from_array(s, "d2", D2)
    out = np.expand_dims(d2, 1)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), np.expand_dims(D2, 1))
    assert out.shape == (None, 1, 3)
    np.testing.assert_array_equal(np.asarray(s.materialize(np.expand_dims(d2, -1))), np.expand_dims(D2, -1))
    with pytest.raises(GraphedTypeError):
        np.expand_dims(d2, 0)  # would displace the partitioned axis


def test_swapaxes_and_transpose_keep_axis0_in_place() -> None:
    s = _s()
    d3 = from_array(s, "d3", D3)
    out = d3.swapaxes(1, 2)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), D3.swapaxes(1, 2))
    assert out.shape == (None, 4, 3)
    assert np.swapaxes(d3, 1, 2).node_id == out.node_id
    tr = d3.transpose(0, 2, 1)
    np.testing.assert_array_equal(np.asarray(s.materialize(tr)), D3.transpose(0, 2, 1))
    assert np.transpose(d3, (0, 2, 1)).node_id == tr.node_id
    with pytest.raises(GraphedTypeError):
        d3.swapaxes(0, 1)
    with pytest.raises(GraphedTypeError):
        d3.transpose(1, 0, 2)
    d2 = from_array(s, "d2", D2)
    with pytest.raises(GraphedTypeError):
        _ = d2.T  # reversing axes moves axis 0 for ndim >= 2


def test_transpose_of_1d_is_identity() -> None:
    s = _s()
    a = from_array(s, "a", V)
    np.testing.assert_array_equal(np.asarray(s.materialize(a.T)), V)
    np.testing.assert_array_equal(np.asarray(s.materialize(np.transpose(a))), V.T)


def test_astype_clip_round_match_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    cast = a.astype(np.float32)
    np.testing.assert_array_equal(np.asarray(s.materialize(cast)), V.astype(np.float32))
    assert cast.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(np.asarray(s.materialize(a.clip(1.5, 6.0))), V.clip(1.5, 6.0))
    assert np.clip(a, 1.5, 6.0).node_id == a.clip(1.5, 6.0).node_id
    np.testing.assert_array_equal(np.asarray(s.materialize(a.clip(lo=2.0))), V.clip(2.0, None))
    np.testing.assert_array_equal(np.asarray(s.materialize(np.round(a, 1))), V.round(1))
    np.testing.assert_array_equal(np.asarray(s.materialize(a.round())), V.round())


def test_take_matches_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    d2 = from_array(s, "d2", D2)
    idx = np.array([0, 5, 5, 2, 11])
    g_idx = from_array(s, "idx", idx)
    np.testing.assert_array_equal(np.asarray(s.materialize(np.take(a, g_idx))), np.take(V, idx))
    inner = np.array([2, 0])
    g_inner = from_array(s, "inner", inner)
    np.testing.assert_array_equal(
        np.asarray(s.materialize(np.take(d2, g_inner, axis=1))), np.take(D2, inner, axis=1)
    )


def test_where_with_arrays_and_scalars_matches_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    b = from_array(s, "b", W)
    cond = a > 4.0
    np.testing.assert_array_equal(np.asarray(s.materialize(np.where(cond, a, b))), np.where(V > 4.0, V, W))
    np.testing.assert_array_equal(
        np.asarray(s.materialize(np.where(cond, a, 0.0))), np.where(V > 4.0, V, 0.0)
    )
    np.testing.assert_array_equal(
        np.asarray(s.materialize(np.where(cond, 1.0, b))), np.where(V > 4.0, 1.0, W)
    )
    np.testing.assert_array_equal(
        np.asarray(s.materialize(np.where(cond, 1.0, 0.0))), np.where(V > 4.0, 1.0, 0.0)
    )


def test_diff_isin_searchsorted_match_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    b = from_array(s, "b", W)
    d2 = from_array(s, "d2", D2)
    np.testing.assert_array_equal(np.asarray(s.materialize(np.diff(a))), np.diff(V))
    np.testing.assert_array_equal(np.asarray(s.materialize(np.diff(a, 2))), np.diff(V, 2))
    np.testing.assert_array_equal(np.asarray(s.materialize(np.diff(d2, axis=0))), np.diff(D2, axis=0))
    np.testing.assert_array_equal(np.asarray(s.materialize(np.isin(a, b))), np.isin(V, W))
    srt = from_array(s, "srt", np.sort(V))
    np.testing.assert_array_equal(
        np.asarray(s.materialize(np.searchsorted(srt, b))), np.searchsorted(np.sort(V), W)
    )
    np.testing.assert_array_equal(
        np.asarray(s.materialize(np.searchsorted(srt, b, side="right"))),
        np.searchsorted(np.sort(V), W, side="right"),
    )


def test_stack_and_vstack_are_refused_in_the_partitioned_mvp() -> None:
    # both would create an inner unknown-length dimension — unrepresentable until Phase 2 N-D chunks
    s = _s()
    a = from_array(s, "a", V)
    b = from_array(s, "b", W)
    with pytest.raises(TypeError):
        np.stack([a, b])
    with pytest.raises(TypeError):
        np.vstack([a, b])
