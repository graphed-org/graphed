"""M13: indexing parity — axis-0 slices/ints (common surface) + numpy tuple subscripts (P2.6).

Slices and integer indexing record boundary nodes on the base proxy (pinned in graphed's M13);
HERE the pins are evaluation (numpy's own semantics) and the numpy-only tuple-subscript idiom:
``a[:, inner...]`` is partition-local (the first element must be the full slice ``:`` — anything
that indexes the partitioned axis 0 inside a tuple is refused in the axis-0-partitioned MVP).
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import Session
from m13_helpers import recorded

from graphed_numpy import NumpyBackend, from_array

V = np.array([5.0, 1.0, 4.0, 2.0, 8.0, 6.0, 3.0, 7.0, 9.0, 0.0, 2.5, 1.5])
D2 = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0], [3.0, 5.0, 8.0]])
D3 = np.arange(24.0).reshape(2, 3, 4)


def _s() -> Session:
    return Session(NumpyBackend())


@pytest.mark.parametrize(
    "key", [slice(2, 9), slice(None, None, 3), slice(5, None), slice(-4, None), slice(2, 11, 2)]
)
def test_axis0_slices_match_numpy(key: slice) -> None:
    s = _s()
    a = from_array(s, "a", V)
    np.testing.assert_array_equal(np.asarray(s.materialize(a[key])), V[key])


def test_integer_index_matches_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    assert float(np.asarray(s.materialize(a[3]))) == V[3]
    assert float(np.asarray(s.materialize(a[-1]))) == V[-1]


def test_boolean_mask_getitem_matches_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    np.testing.assert_array_equal(np.asarray(s.materialize(a[a > 4.0])), V[V > 4.0])


def test_fancy_integer_getitem_matches_numpy() -> None:
    s = _s()
    a = from_array(s, "a", V)
    idx = np.array([0, 5, 5, 2, 11])
    g_idx = from_array(s, "idx", idx)
    np.testing.assert_array_equal(np.asarray(s.materialize(a[g_idx])), V[idx])


def test_tuple_subscripts_match_numpy_and_stay_fusible() -> None:
    s = _s()
    d2 = from_array(s, "d2", D2)
    d3 = from_array(s, "d3", D3)
    np.testing.assert_array_equal(np.asarray(s.materialize(d2[:, 1])), D2[:, 1])
    np.testing.assert_array_equal(np.asarray(s.materialize(d2[:, 0:2])), D2[:, 0:2])
    np.testing.assert_array_equal(np.asarray(s.materialize(d3[:, 1, :])), D3[:, 1, :])
    np.testing.assert_array_equal(np.asarray(s.materialize(d3[:, ::2, 1])), D3[:, ::2, 1])
    node = recorded(s, d2[:, 1])
    assert node["kind"] == "op"  # partition-local: fusible
    assert node["name"] == "subscript"
    assert d2[:, 1].shape == (None,)
    assert d2[:, 0:2].shape == (None, 2)


def test_tuple_subscripts_interning() -> None:
    s = _s()
    d2 = from_array(s, "d2", D2)
    assert d2[:, 1].node_id == d2[:, 1].node_id
    assert d2[:, 1].node_id != d2[:, 2].node_id


def test_indexing_the_partitioned_axis_inside_a_tuple_is_refused() -> None:
    s = _s()
    d2 = from_array(s, "d2", D2)
    with pytest.raises(TypeError):
        _ = d2[1, :]  # axis 0 must stay the full slice in the axis-0-partitioned MVP
    with pytest.raises(TypeError):
        _ = d2[1:3, :]
    with pytest.raises(TypeError):
        _ = d2[:, 1.5]
