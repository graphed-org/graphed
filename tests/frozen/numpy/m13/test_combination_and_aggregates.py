"""M13: cross-array combination + tree-reducible analytics (dask.array parity P2.7).

Concatenation along the partitioned axis RESTRUCTURES the partition set, so it records a
boundary node; along inner axes it is partition-local and fusible. The analytics that consume
the whole partitioned axis (unique/bincount/histogram*) are boundary reductions. ``histogram``
keeps numpy's (counts, edges) contract: the counts are deferred, the edges — fully determined by
``bins``/``range`` at record time — are concrete.
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import GraphedTypeError, Session
from m13_helpers import recorded

from graphed_numpy import NumpyBackend, from_array

V = np.array([5.0, 1.0, 4.0, 2.0, 8.0, 6.0, 3.0, 7.0, 9.0, 0.0, 2.5, 1.5])
W = np.array([2.0, 0.5, 3.5, 1.0, 7.5, 6.5, 2.5, 8.0, 9.5, 0.5, 3.0, 1.0])
A2 = np.array([[3.0, 1.0], [1.0, 5.0], [2.0, 6.0]])
B2 = np.array([[4.0, 9.0, 5.0], [8.0, 3.0, 5.0], [7.0, 0.0, 2.0]])


def _s() -> Session:
    return Session(NumpyBackend())


def test_concatenate_axis0_is_a_boundary_node() -> None:
    s = _s()
    a = from_array(s, "a", V)
    b = from_array(s, "b", W)
    out = np.concatenate([a, b])
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), np.concatenate([V, W]))
    node = recorded(s, out)
    assert node["kind"] == "reduction"
    assert node["name"] == "concatenate"
    assert out.shape == (None,)


def test_concatenate_inner_axis_is_fusible() -> None:
    s = _s()
    a = from_array(s, "a", A2)
    b = from_array(s, "b", B2)
    out = np.concatenate([a, b], axis=1)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), np.concatenate([A2, B2], axis=1))
    assert recorded(s, out)["kind"] == "op"
    assert out.shape == (None, 5)


def test_hstack_interns_with_concatenate() -> None:
    s = _s()
    a = from_array(s, "a", V)
    b = from_array(s, "b", W)
    assert np.hstack([a, b]).node_id == np.concatenate([a, b]).node_id  # 1-D: axis 0
    a2 = from_array(s, "a2", A2)
    b2 = from_array(s, "b2", B2)
    assert np.hstack([a2, b2]).node_id == np.concatenate([a2, b2], axis=1).node_id  # >=2-D: axis 1
    np.testing.assert_array_equal(np.asarray(s.materialize(np.hstack([a, b]))), np.hstack([V, W]))


def test_unique_matches_numpy_and_is_a_reduction() -> None:
    s = _s()
    a = from_array(s, "a", np.array([3.0, 1.0, 3.0, 2.0, 1.0, 9.0]))
    out = np.unique(a)
    np.testing.assert_array_equal(
        np.asarray(s.materialize(out)), np.unique(np.array([3.0, 1.0, 3.0, 2.0, 1.0, 9.0]))
    )
    assert recorded(s, out)["kind"] == "reduction"
    assert out.shape == (None,)  # the number of distinct values is data-dependent


def test_bincount_matches_numpy_and_rejects_floats() -> None:
    s = _s()
    ints = np.array([1, 3, 1, 0, 5, 3, 3])
    a = from_array(s, "a", ints)
    out = np.bincount(a)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), np.bincount(ints))
    assert recorded(s, out)["kind"] == "reduction"
    floats = from_array(s, "f", V)
    with pytest.raises(GraphedTypeError):
        np.bincount(floats)


def test_histogram_keeps_the_counts_edges_contract() -> None:
    s = _s()
    a = from_array(s, "a", V)
    counts, edges = np.histogram(a, bins=6, range=(0.0, 9.0))
    ref_counts, ref_edges = np.histogram(V, bins=6, range=(0.0, 9.0))
    np.testing.assert_array_equal(np.asarray(s.materialize(counts)), ref_counts)
    assert isinstance(edges, np.ndarray)  # edges are record-time-concrete
    np.testing.assert_allclose(edges, ref_edges)
    node = recorded(s, counts)
    assert node["kind"] == "reduction"
    assert counts.shape == (6,)  # the partitioned axis is consumed: concrete output
    with pytest.raises(TypeError):
        np.histogram(a, bins=6)  # without a range the edges cannot be known at record time


def test_histogram2d_and_histogramdd_match_numpy() -> None:
    s = _s()
    x = from_array(s, "x", V)
    y = from_array(s, "y", W)
    counts, xe, ye = np.histogram2d(x, y, bins=5, range=[(0.0, 10.0), (0.0, 10.0)])
    ref, rxe, rye = np.histogram2d(V, W, bins=5, range=[(0.0, 10.0), (0.0, 10.0)])
    np.testing.assert_array_equal(np.asarray(s.materialize(counts)), ref)
    np.testing.assert_allclose(xe, rxe)
    np.testing.assert_allclose(ye, rye)
    assert counts.shape == (5, 5)

    sample = np.column_stack([V, W, V + W])
    d = from_array(s, "d", sample)
    dd, _ = np.histogramdd(d, bins=3, range=[(0.0, 20.0)] * 3)
    ref_dd, _ = np.histogramdd(sample, bins=3, range=[(0.0, 20.0)] * 3)
    np.testing.assert_array_equal(np.asarray(s.materialize(dd)), ref_dd)
    assert dd.shape == (3, 3, 3)


def _ir() -> bytes:
    s = _s()
    a = from_array(s, "a", V)
    b = from_array(s, "b", W)
    counts, _ = np.histogram(np.where(a > 4.0, a, b).clip(0.0, 9.0), bins=4, range=(0.0, 9.0))
    return s.serialized_ir(counts)


def test_manipulation_programs_are_deterministic() -> None:
    assert _ir() == _ir()
