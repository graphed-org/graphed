"""Draws from equal generators that differ in their arguments are different sources."""

from __future__ import annotations

import numpy as np

import graphed
from graphed.numpy import NumpyBackend, default_rng


def test_a_draw_with_other_parameters_is_its_own_source() -> None:
    session = graphed.Session(NumpyBackend())
    a = default_rng(session, 7).normal(3)
    b = default_rng(session, 7).normal(3, loc=100.0)
    va, vb = graphed.materialize([a, b])
    assert a.node_id != b.node_id
    assert abs(np.mean(va)) < 10
    assert abs(np.mean(vb) - 100.0) < 10


def test_a_draw_of_another_shape_is_its_own_source() -> None:
    session = graphed.Session(NumpyBackend())
    a = default_rng(session, 7).uniform(3)
    b = default_rng(session, 7).uniform(5)
    va, vb = graphed.materialize([a, b])
    assert (va.shape, vb.shape) == ((3,), (5,))


def test_the_same_draw_from_equal_generators_is_one_source() -> None:
    session = graphed.Session(NumpyBackend())
    a = default_rng(session, 7).integers(4, 0, 10)
    b = default_rng(session, 7).integers(4, 0, 10)
    assert a.node_id == b.node_id
