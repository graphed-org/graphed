"""Tree-reducible reduction monoids (M12, dask.array parity P1.4).

``monoid(kind)`` is the (chunk, combine, empty, finalize) quadruple that drops straight into the
M7 process/combine/empty execution model — dask's generic ``da.reduction(chunk, combine,
aggregate)`` is exactly this triple plus the aggregate. ``combine`` must be associative with
``empty()`` as identity, so ANY chunking and ANY combine-tree shape (left fold, balanced tree,
straggler-driven order) yields the same finalized value as the whole-array numpy reduction.

mean/var/std are carried as moment sums ``(count, total, total-of-squares)`` — exactly
tree-mergeable; ``finalize`` turns the state into the numpy-equivalent statistic. Kinds that are
NOT tree-reducible (e.g. median) are refused rather than approximated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

State = Any


@dataclass(frozen=True)
class Monoid:
    """A tree-reducible reduction: ``finalize(combine(chunk(x), chunk(y), ...))``."""

    chunk: Callable[[Any], State]
    combine: Callable[[State, State], State]
    empty: Callable[[], State]
    finalize: Callable[[State], Any]


def _identity(state: State) -> Any:
    return state


def _add(a: State, b: State) -> State:
    return a + b


def _zero() -> State:
    return 0.0


def _one() -> State:
    return 1.0


def _sum_chunk(x: Any) -> State:
    return float(np.sum(x))


def _prod_chunk(x: Any) -> State:
    return float(np.prod(x))


def _mul(a: State, b: State) -> State:
    return a * b


def _min_chunk(x: Any) -> State:
    return float(np.min(x))


def _max_chunk(x: Any) -> State:
    return float(np.max(x))


def _min2(a: State, b: State) -> State:
    return min(a, b)


def _max2(a: State, b: State) -> State:
    return max(a, b)


def _pos_inf() -> State:
    return math.inf


def _neg_inf() -> State:
    return -math.inf


def _count_chunk(x: Any) -> State:
    return int(np.size(x))


def _int_zero() -> State:
    return 0


def _any_chunk(x: Any) -> State:
    return bool(np.any(x))


def _all_chunk(x: Any) -> State:
    return bool(np.all(x))


def _or(a: State, b: State) -> State:
    return bool(a or b)


def _and(a: State, b: State) -> State:
    return bool(a and b)


def _true() -> State:
    return True


def _false() -> State:
    return False


def _moments_chunk(x: Any) -> State:
    a = np.asarray(x, dtype=float)
    return (int(a.size), float(a.sum()), float((a * a).sum()))


def _moments_combine(a: State, b: State) -> State:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _moments_empty() -> State:
    return (0, 0.0, 0.0)


def _mean_finalize(state: State) -> float:
    n, total, _ = state
    return float(total / n) if n else math.nan


def _var_finalize(ddof: int) -> Callable[[State], float]:
    def finalize(state: State) -> float:
        n, total, total2 = state
        if n - ddof <= 0:
            return math.nan
        return float((total2 - total * total / n) / (n - ddof))

    return finalize


def _std_finalize(ddof: int) -> Callable[[State], float]:
    var = _var_finalize(ddof)

    def finalize(state: State) -> float:
        return math.sqrt(var(state))

    return finalize


def monoid(kind: str, *, ddof: int = 0) -> Monoid:
    """The tree-reducible monoid for ``kind`` (``ValueError`` if no such monoid exists)."""
    if kind == "sum":
        return Monoid(_sum_chunk, _add, _zero, _identity)
    if kind == "prod":
        return Monoid(_prod_chunk, _mul, _one, _identity)
    if kind == "min":
        return Monoid(_min_chunk, _min2, _pos_inf, _identity)
    if kind == "max":
        return Monoid(_max_chunk, _max2, _neg_inf, _identity)
    if kind == "count":
        return Monoid(_count_chunk, _add, _int_zero, _identity)
    if kind == "any":
        return Monoid(_any_chunk, _or, _false, _identity)
    if kind == "all":
        return Monoid(_all_chunk, _and, _true, _identity)
    if kind == "mean":
        return Monoid(_moments_chunk, _moments_combine, _moments_empty, _mean_finalize)
    if kind == "var":
        return Monoid(_moments_chunk, _moments_combine, _moments_empty, _var_finalize(ddof))
    if kind == "std":
        return Monoid(_moments_chunk, _moments_combine, _moments_empty, _std_finalize(ddof))
    raise ValueError(f"no tree-reducible monoid for {kind!r}")
