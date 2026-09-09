"""m56 §2 item 5 / §4 downstream: the m53 knobs see the cross-kind joints like any other. The union
flag collapses them, a placement keeps one by name and prunes the rest, and the guard's bound is the
product of family sizes — this family times EACH foreign family, nominal included.
"""

from __future__ import annotations

import pytest
from m56_fanout_fixtures import (
    DESIGN_BOUND,
    PRE_M56_BOUND,
    UNION_LABELS,
    m56_capstone,
    m56_joints,
    m56_minted,
)

import graphed
from graphed.errors import GraphedError

ALL_JOINTS = m56_joints("hf", "jes") | m56_joints("hf", "jer")


def test_composes_as_union_collapses_the_cross_kind_joints() -> None:
    assert m56_minted(m56_capstone().weight, "hf") == ALL_JOINTS  # the flag has joints to collapse

    unioned = m56_capstone(composes_as_union=True).weight
    assert set(graphed.labels(unioned)) == set(UNION_LABELS)


def test_a_placement_keeps_a_named_cross_kind_joint_and_prunes_the_rest() -> None:
    weight = m56_capstone(placements=[{"hf": "up", "jes": "up"}]).weight

    assert m56_minted(weight, "hf") == {"hf_up__jes_up"}
    assert set(graphed.labels(weight)) == set(UNION_LABELS) | {"hf_up__jes_up"}


def test_the_guard_bounds_the_grid_over_every_foreign_family() -> None:
    with pytest.raises(GraphedError) as caught:
        m56_capstone(max_universes=PRE_M56_BOUND)

    message = str(caught.value)
    assert str(DESIGN_BOUND) in message  # jer(3) x jes(3) x hf(3), not jer(3) x hf(3)
    assert "jes" in message and "jer" in message and "hf" in message

    admitted = m56_capstone(max_universes=DESIGN_BOUND).weight
    assert set(graphed.labels(admitted)) == set(UNION_LABELS) | ALL_JOINTS
