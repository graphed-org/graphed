"""M12: deterministic random sources + tree-reducible reduction monoids (parity P1.4/P1.5).

Random: a seeded generator records ordinary sources whose values — and whose serialized IR — are
byte-identical across runs for the same seed (the determinism gate extends to random programs).

Monoids: `graphed.numpy.monoid(kind)` is the (chunk, combine, empty, finalize) quadruple that
drops into the M7 process/combine/empty execution model (dask's generic `reduction()` is exactly
this triple + aggregate). Combining over ANY chunking/tree shape must agree with numpy whole-array.
"""

from __future__ import annotations

from functools import reduce as fold

import numpy as np
import pytest

import graphed.numpy as gn
from graphed import Session

# ---- random ---------------------------------------------------------------------------------------


def test_same_seed_reproduces_values_and_ir() -> None:
    def build() -> tuple[bytes, np.ndarray]:
        s = Session(gn.NumpyBackend())
        rng = gn.default_rng(s, seed=42)
        a = rng.normal(8)
        return s.serialized_ir(a.sum()), np.asarray(s.materialize(a))

    ir1, v1 = build()
    ir2, v2 = build()
    assert ir1 == ir2
    np.testing.assert_array_equal(v1, v2)


def test_different_seeds_and_draws_differ() -> None:
    s = Session(gn.NumpyBackend())
    a = np.asarray(s.materialize(gn.default_rng(s, seed=1).normal(16)))
    b = np.asarray(s.materialize(gn.default_rng(s, seed=2).normal(16)))
    assert not np.array_equal(a, b)
    rng = gn.default_rng(s, seed=3)
    first, second = rng.normal(16), rng.normal(16)
    assert first.node_id != second.node_id  # successive draws are distinct sources
    assert not np.array_equal(np.asarray(s.materialize(first)), np.asarray(s.materialize(second)))


def test_uniform_and_integers_distributions() -> None:
    s = Session(gn.NumpyBackend())
    rng = gn.default_rng(s, seed=7)
    u = np.asarray(s.materialize(rng.uniform(64, low=2.0, high=3.0)))
    assert u.shape == (64,) and (u >= 2.0).all() and (u < 3.0).all()
    i = np.asarray(s.materialize(rng.integers(64, low=0, high=10)))
    assert np.issubdtype(i.dtype, np.integer) and (i >= 0).all() and (i < 10).all()
    n = np.asarray(s.materialize(rng.standard_normal((4, 3))))
    assert n.shape == (4, 3)


# ---- monoids --------------------------------------------------------------------------------------

DATA = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0])
CHUNKINGS = [[11], [4, 7], [1, 2, 3, 5], [2, 2, 2, 2, 2, 1]]

REFERENCE = {
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "var": np.var,
    "std": np.std,
    "count": np.size,
    "any": np.any,
    "all": np.all,
}


def _chunks(data: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    out, start = [], 0
    for n in sizes:
        out.append(data[start : start + n])
        start += n
    assert start == len(data)
    return out


@pytest.mark.parametrize("kind", sorted(REFERENCE))
@pytest.mark.parametrize("sizes", CHUNKINGS, ids=[str(c) for c in CHUNKINGS])
def test_monoid_agrees_with_numpy_over_any_chunking(kind: str, sizes: list[int]) -> None:
    m = gn.monoid(kind)
    data = (DATA > 3.0) if kind in ("any", "all") else DATA
    states = [m.chunk(c) for c in _chunks(data, sizes)]
    # left fold AND balanced tree must agree (associativity over the tree shape)
    left = fold(m.combine, states, m.empty())
    while len(states) > 1:
        states = [
            m.combine(*states[i : i + 2]) if i + 1 < len(states) else states[i]
            for i in range(0, len(states), 2)
        ]
    assert m.finalize(left) == pytest.approx(m.finalize(states[0]))
    assert m.finalize(left) == pytest.approx(REFERENCE[kind](data))


def test_var_monoid_supports_ddof() -> None:
    m = gn.monoid("var", ddof=1)
    states = [m.chunk(c) for c in _chunks(DATA, [4, 7])]
    assert m.finalize(m.combine(*states)) == pytest.approx(np.var(DATA, ddof=1))


def test_empty_is_the_identity() -> None:
    for kind in sorted(REFERENCE):
        m = gn.monoid(kind)
        state = m.chunk((DATA > 3.0) if kind in ("any", "all") else DATA)
        assert m.finalize(m.combine(state, m.empty())) == pytest.approx(m.finalize(state))
        assert m.finalize(m.combine(m.empty(), state)) == pytest.approx(m.finalize(state))


def test_unknown_monoid_kind_raises() -> None:
    with pytest.raises(ValueError):
        gn.monoid("median")  # median is not tree-reducible; it must not pretend to be
