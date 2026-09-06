"""vary-m53 §4: the loud guard. An un-selected default grid larger than `max_universes` raises,
naming the count and the families; at or below the budget it is silent; and neither an explicit
`points=` selection nor `composes_as_union=True` is ever guarded."""

from __future__ import annotations

import pytest
from m53_policy_fixtures import UNION_LABELS, fanout_weight

import graphed
from graphed.errors import GraphedError


def test_the_guard_raises_above_the_budget_naming_the_count_and_families() -> None:
    with pytest.raises(GraphedError) as caught:
        fanout_weight(max_universes=14)  # the grid is 15 = jes(3) x btag(5)

    message = str(caught.value)
    assert "15" in message  # the surfaced count
    assert "jes" in message and "btag" in message  # the families that produced it


def test_the_guard_is_silent_at_the_budget() -> None:
    _session, _registered, weight = fanout_weight(max_universes=15)
    assert len(graphed.labels(weight)) == 15  # minted, no raise


def test_a_raised_budget_lets_a_grid_through_that_a_low_one_rejects() -> None:
    with pytest.raises(GraphedError):
        fanout_weight(max_universes=8)  # 15 > 8

    _session, _registered, weight = fanout_weight(max_universes=64)  # the same grid, higher budget
    assert len(graphed.labels(weight)) == 15


def test_the_default_budget_admits_the_benchmark_grid() -> None:
    _session, _registered, weight = fanout_weight()  # no max_universes → the default admits 15
    assert len(graphed.labels(weight)) == 15


def test_a_selection_and_a_union_collapse_are_never_guarded() -> None:
    # an explicit points= selection is the analyst's own enumeration — a tiny budget cannot block it
    _s1, _r1, pruned = fanout_weight(max_universes=1, points=[{"btag": "hf_up", "jes": "up"}])
    assert "btag_hf_up__jes_up" in graphed.labels(pruned)

    # composes_as_union collapses the fanout away, so there is nothing to bound
    _s2, _r2, unioned = fanout_weight(max_universes=1, composes_as_union=True)
    assert set(graphed.labels(unioned)) == set(UNION_LABELS)
