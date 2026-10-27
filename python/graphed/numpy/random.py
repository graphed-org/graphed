"""Deterministic random sources (M12, dask.array parity P1.5).

A ``GraphedRNG`` records ordinary named sources whose data comes from numpy's ``default_rng``
seeded by ``(seed, draw-counter)`` and whose NAME encodes the same pair — so for the same seed and
the same program, both the values and the serialized IR are byte-identical across runs (the M8
determinism gate extends to random programs), while successive draws are distinct sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast, overload

import numpy as np

from graphed import Array, Session
from graphed.session import _reader

from . import NumpyBackend
from .creation import ShapeLike, _norm_shape, _source


@dataclass
class GraphedRNG:
    """A deterministic generator of random source arrays for ``session``."""

    session: Session
    seed: int
    _draws: int = field(default=0, init=False)

    def _draw(self, kind: str, sampler: Any) -> Array:
        self._draws += 1
        rng = np.random.default_rng([self.seed, self._draws])
        arr = np.asarray(sampler(rng))
        name = f"random[{kind},seed={self.seed},draw={self._draws}]"
        return _source(self.session, None, name, arr)

    def normal(self, size: ShapeLike, loc: float = 0.0, scale: float = 1.0) -> Array:
        """Create a source array of normal draws."""
        shape = _norm_shape(size)
        return self._draw("normal", lambda r: r.normal(loc, scale, shape))

    def standard_normal(self, size: ShapeLike) -> Array:
        """Create a source array of standard-normal draws."""
        shape = _norm_shape(size)
        return self._draw("standard_normal", lambda r: r.standard_normal(shape))

    def uniform(self, size: ShapeLike, low: float = 0.0, high: float = 1.0) -> Array:
        """Create a source array of uniform draws in ``[low, high)``."""
        shape = _norm_shape(size)
        return self._draw("uniform", lambda r: r.uniform(low, high, shape))

    def integers(self, size: ShapeLike, low: int, high: int) -> Array:
        """Create a source array of integer draws in ``[low, high)``."""
        shape = _norm_shape(size)
        return self._draw("integers", lambda r: r.integers(low, high, shape))


@overload
def default_rng(session: Session, seed: int) -> GraphedRNG: ...
@overload
def default_rng(seed: int, *, session: Session | None = None) -> GraphedRNG: ...
@_reader(NumpyBackend)
def default_rng(seed: int, *, session: Session | None = None) -> GraphedRNG:
    """Return a deterministic generator of random source arrays for ``session``.

    Else for the current Session.
    """
    return GraphedRNG(cast("Session", session), int(seed))
