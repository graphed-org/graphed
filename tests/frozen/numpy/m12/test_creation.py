"""M12: creation routines (dask.array parity P1.5).

Concrete creators (`zeros/ones/full/empty/arange/linspace`) record deterministic named sources —
identical creations intern to the SAME source node. `*_like` creators are recorded as fusible ops
(the operand's length along the partitioned axis is unknown at record time). `empty`/`empty_like`
are deterministic: they are zeros (uninitialized memory would break the byte-identity gate).
"""

from __future__ import annotations

import numpy as np
from graphed import Session

import graphed_numpy as gn


def test_concrete_creators_match_numpy() -> None:
    s = Session(gn.NumpyBackend())
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.zeros(s, (3, 2)))), np.zeros((3, 2)))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.ones(s, 4))), np.ones(4))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.full(s, 3, 7.5))), np.full(3, 7.5))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.empty(s, 5))), np.zeros(5))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.arange(s, 2, 11, 3))), np.arange(2, 11, 3))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.arange(s, 5))), np.arange(5))
    np.testing.assert_array_equal(
        np.asarray(s.materialize(gn.linspace(s, 0.0, 1.0, 5))), np.linspace(0.0, 1.0, 5)
    )


def test_creators_respect_dtype() -> None:
    s = Session(gn.NumpyBackend())
    assert gn.zeros(s, 3, dtype=np.int32).dtype == np.dtype(np.int32)
    assert gn.arange(s, 4).dtype == np.arange(4).dtype
    assert gn.ones(s, 3).dtype == np.dtype(np.float64)


def test_identical_creations_intern_to_one_source() -> None:
    s = Session(gn.NumpyBackend())
    a = gn.zeros(s, (3, 2))
    n = s.node_count()
    b = gn.zeros(s, (3, 2))
    assert b.node_id == a.node_id
    assert s.node_count() == n
    assert gn.zeros(s, (3, 3)).node_id != a.node_id  # different content, different source


def test_like_creators_record_fusible_ops_and_match_numpy() -> None:
    s = Session(gn.NumpyBackend())
    a = gn.from_array(s, "a", np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.zeros_like(a))), np.zeros((2, 2)))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.ones_like(a))), np.ones((2, 2)))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.full_like(a, 9.0))), np.full((2, 2), 9.0))
    np.testing.assert_array_equal(np.asarray(s.materialize(gn.empty_like(a))), np.zeros((2, 2)))
    assert gn.zeros_like(a).shape == (None, 2)  # partitioned axis preserved: it is an OP
    assert gn.zeros_like(a).dtype == np.dtype(np.float64)


def test_from_array_accepts_chunk_metadata() -> None:
    s = Session(gn.NumpyBackend())
    data = np.arange(8, dtype=np.float64)
    a = gn.from_array(s, "a", data, chunks=4)
    np.testing.assert_array_equal(np.asarray(s.materialize(a)), data)
    b = gn.from_array(s, "b", data)  # chunks are metadata only: evaluation unchanged
    np.testing.assert_array_equal(np.asarray(s.materialize(a)), np.asarray(s.materialize(b)))


def _ir() -> bytes:
    s = Session(gn.NumpyBackend())
    out = (gn.arange(s, 6) * gn.ones(s, 6) + gn.full(s, 6, 2.0)).sum()
    return s.serialized_ir(out)


def test_creation_is_deterministic() -> None:
    assert _ir() == _ir()
