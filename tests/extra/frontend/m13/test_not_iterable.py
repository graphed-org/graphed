"""Implementer regression (M13 iteration 1): int __getitem__ must not make Array iterable.

Python's legacy iteration protocol falls back to a[0], a[1], ... which on a deferred graph never
raises IndexError — np.concatenate(a) (or any tuple(a)) would record nodes forever. The explicit
__iter__ refusal turns that into an immediate TypeError.
"""

from __future__ import annotations

import os
import sys

import pytest

from graphed import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "frozen", "m13"))
from m13_toy import ToyBackend, source


def test_deferred_arrays_are_not_iterable() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        iter(a)
    with pytest.raises(TypeError):
        list(a)
    n = s.node_count()
    with pytest.raises(TypeError):
        tuple(a)
    assert s.node_count() == n  # the failed iteration recorded NOTHING
