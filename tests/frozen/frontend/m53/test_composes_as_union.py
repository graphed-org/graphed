"""vary-m53 §2.5: `composes_as_union=True` is the one-flag collapse back to the pre-m53 datacard
union, and it is mutually exclusive with an explicit `points=` selection."""

from __future__ import annotations

import pytest
from m53_policy_fixtures import UNION_LABELS, fanout_weight

import graphed
from graphed.errors import GraphedError


def test_composes_as_union_collapses_a_dependent_family_to_the_union() -> None:
    _session, _registered, weight = fanout_weight(composes_as_union=True)

    labels = graphed.labels(weight)
    assert set(labels) == set(UNION_LABELS)  # the seven one-at-a-time universes, no joints
    assert not [label for label in labels if "__" in label]


def test_composes_as_union_with_points_is_a_construction_error() -> None:
    with pytest.raises(GraphedError) as caught:
        fanout_weight(composes_as_union=True, points=[{"btag": "hf_up", "jes": "up"}])
    assert "composes_as_union" in str(caught.value)  # names the incompatible combination, not a bad member
