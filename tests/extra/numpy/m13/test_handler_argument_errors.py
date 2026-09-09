"""Implementer-side coverage for the M13 handler argument-validation branches.

The frozen suites pin the functional behavior; these exercise the loud-failure paths of the
numpy-function handlers (wrong arities, unsupported kwargs, non-deferred operands) so every
refusal is executed, not just written.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from graphed import Array, Session
from graphed.numpy import NumpyBackend, from_array

# The calls under `pytest.raises(TypeError)` are deliberately ill-typed — refusing them is what
# they assert. numpy's stubs disagree about which ones are static errors across versions (1.x on
# py3.11, 2.x on py3.12+), so they go through an untyped alias rather than suppressions that are
# required on one generation and unused on the other.
_np: Any = np

V = np.array([1.0, 2.0, 3.0, 4.0])
IDX = np.array([0, 2])


@pytest.fixture
def a() -> Array:
    return from_array(Session(NumpyBackend()), "a", V)


def test_reduction_and_scan_handler_argument_errors(a: Array) -> None:
    with pytest.raises(TypeError):
        np.sum(a, 0, 1)  # too many positionals
    with pytest.raises(TypeError):
        _np.sum(a, 0, axis=1)  # positional + keyword axis
    with pytest.raises(TypeError):
        _np.sum(a, axis=1.5)
    with pytest.raises(TypeError):
        np.sum(a, out=np.empty(1))
    with pytest.raises(TypeError):
        np.cumsum(a, dtype=np.float32)
    with pytest.raises(TypeError):
        np.cumsum(a, axis=True)


def test_manipulation_handler_argument_errors(a: Array) -> None:
    s = a.session
    with pytest.raises(TypeError):
        np.reshape(a, (-1, 2), order="F")
    with pytest.raises(TypeError):
        _np.squeeze(a, 1, 2)
    with pytest.raises(TypeError):
        _np.transpose(a, (0,), axes=(0,))
    with pytest.raises(TypeError):
        np.transpose(a, 7)
    with pytest.raises(TypeError):
        _np.swapaxes(a, 1)
    with pytest.raises(TypeError):
        _np.expand_dims(a, 1.5)
    with pytest.raises(TypeError):
        np.clip(a, 1.0, 2.0, out=np.empty(1))
    with pytest.raises(TypeError):
        _np.round(a, 1, 2)
    with pytest.raises(TypeError):
        np.take(a, np.array([0]))  # indices must be deferred
    with pytest.raises(TypeError):
        np.take(a, from_array(s, "i", IDX), axis=0, mode="wrap")
    with pytest.raises(TypeError):
        a.clip()
    with pytest.raises(TypeError):
        a.reshape(2.5)
    with pytest.raises(TypeError):
        _ = a["x":"y"]


def test_combination_handler_argument_errors(a: Array) -> None:
    s = a.session
    b = from_array(s, "b", V)
    with pytest.raises(TypeError):
        _np.where(a > 1.0, a, b, 3)
    with pytest.raises(TypeError):
        np.where(np.ones(4) > 0, a, b)  # condition must be deferred
    with pytest.raises(TypeError):
        np.concatenate(a)
    with pytest.raises(TypeError):
        np.concatenate([a, np.ones(4)])
    with pytest.raises(TypeError):
        np.concatenate([a, b], 0, out=np.empty(8))
    with pytest.raises(TypeError):
        np.hstack([a, b], dtype=np.float32)
    with pytest.raises(TypeError):
        np.isin(a, np.ones(4))
    with pytest.raises(TypeError):
        np.searchsorted(a, b, sorter=np.arange(4))
    with pytest.raises(TypeError):
        np.unique(a, return_counts=True)
    with pytest.raises(TypeError):
        np.bincount(a, minlength=10)
    with pytest.raises(TypeError):
        np.diff(a, 1, axis=0, prepend=0.0)


def test_histogram_handler_argument_errors(a: Array) -> None:
    s = a.session
    b = from_array(s, "b", V)
    with pytest.raises(TypeError):
        np.histogram(a, bins=np.array([0.0, 1.0]), range=(0, 1))  # edges-array bins unsupported
    with pytest.raises(TypeError):
        np.histogram(a, 4, range=(0, 1), weights=np.ones(4))
    with pytest.raises(TypeError):
        _np.histogram(a, 4, range=(0,))
    with pytest.raises(TypeError):
        np.histogram2d(a, b, bins=4)  # no range
    with pytest.raises(TypeError):
        np.histogram2d(a, np.ones(4), bins=4, range=[(0, 1), (0, 1)])
    with pytest.raises(TypeError):
        np.histogramdd(a, bins=4)  # no range
    with pytest.raises(TypeError):
        np.histogramdd(a, bins=4.5, range=[(0, 1)])
    with pytest.raises(TypeError):
        _np.histogramdd(a, bins=4, range=7)


def test_unsupported_metadata_and_subscript_branches(a: Array) -> None:
    with pytest.raises(TypeError):
        _ = a[:, "x"]
    with pytest.raises(TypeError):
        _ = a[:, 1.5:2.5]
    with pytest.raises(TypeError):
        a.transpose(1.5)
