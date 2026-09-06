"""vary-m53 §3: `points=` PRUNES the auto grid to a named subset and is a PRECISION knob for custom
coordinates — validated by reachability, so an unreachable point is refused loudly."""

from __future__ import annotations

import pytest
from m53_policy_fixtures import UNION_LABELS, fanout_weight, numeric_fanout

import graphed
from graphed.errors import GraphedError


def test_points_prune_keeps_only_the_named_joints() -> None:
    selection = [{"btag": "hf_up", "jes": "up"}, {"btag": "lf_down", "jes": "down"}]
    _session, _registered, weight = fanout_weight(points=selection)

    labels = set(graphed.labels(weight))
    assert "btag_hf_up__jes_up" in labels
    assert "btag_lf_down__jes_down" in labels

    for pruned in (
        "btag_hf_up__jes_down",
        "btag_hf_down__jes_up",
        "btag_hf_down__jes_down",
        "btag_lf_up__jes_up",
        "btag_lf_up__jes_down",
        "btag_lf_down__jes_up",
    ):
        assert pruned not in labels, pruned

    for base in UNION_LABELS:  # nominal, the jes leak and the one-at-a-time b-tags are untouched
        assert base in labels, base


def test_points_prune_refuses_an_unreachable_point() -> None:
    # 'sideways' is not a registered jes universe, so the point resolves to no member
    with pytest.raises(GraphedError):
        fanout_weight(points=[{"btag": "hf_up", "jes": "sideways"}])


def test_points_precision_accepts_a_reachable_numeric_coordinate() -> None:
    registered = numeric_fanout(points=[{"muF": "2", "jes": 2}])

    reported = graphed.points(graphed.weight(registered))
    joints = [label for label, point in reported.items() if point == {"jes": "2", "muF": "2"}]
    assert len(joints) == 1, reported  # the numeric coordinate resolved to exactly one universe
