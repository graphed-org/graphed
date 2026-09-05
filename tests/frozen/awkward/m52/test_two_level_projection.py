"""C3 / design §4.6, §8-e: `context._two_level` is the same resolution applied at each of its two
levels, and it is where R1-c is earned.

The weight form already RECORDS every cross member inside the nested factor container; today
`_two_level` discards them to produce the diagonal. Projection makes the recorded cross member
reachable by name, so a joint universe costs the interning of nodes that already exist.
"""

from __future__ import annotations

from m52_projection_fixtures import JOINT_LABEL, ONE_AT_A_TIME_LABEL, joint_weight_program

import graphed
from graphed.context import _two_level
from graphed.varied import rebuild


def test_two_level_reaches_the_true_inner_cross_member() -> None:
    program = joint_weight_program()
    factor = rebuild(program.factor_members)

    inner = graphed.member_of(factor, JOINT_LABEL)
    cross = graphed.member_of(inner, "jes_up")
    diagonal = graphed.nominal(inner)
    assert cross.node_id != diagonal.node_id  # the off-diagonal member is a distinct, real node

    assert _two_level(factor, JOINT_LABEL).node_id == cross.node_id

    # the diagonal stays the diagonal: a one-at-a-time label names no `jes` coordinate
    one_at_a_time_inner = graphed.member_of(factor, ONE_AT_A_TIME_LABEL)
    assert (
        _two_level(factor, ONE_AT_A_TIME_LABEL).node_id
        == graphed.nominal(one_at_a_time_inner).node_id
    )
