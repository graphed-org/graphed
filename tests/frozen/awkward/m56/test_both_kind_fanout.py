"""m56 §2 item 1 / §4 headline: a family registered over jets a BOTH-KIND nuisance shifts fans out
over that nuisance as well as over the shift-only one, and each joint resolves to the cross node the
graph holds rather than collapsing onto a one-at-a-time universe.
"""

from __future__ import annotations

from m56_fanout_fixtures import TAGS, UNION_LABELS, m56_capstone, m56_joints, m56_minted

import graphed
from graphed import Kind


def test_a_both_kind_nuisance_fans_out_beside_the_shift_only_one() -> None:
    program = m56_capstone()
    weight = program.weight

    assert graphed.variations(program.ctx)["jes"]["up"][0] == Kind.WEIGHT | Kind.SHIFT
    assert m56_minted(weight, "hf") == m56_joints("hf", "jes") | m56_joints("hf", "jer")
    assert set(graphed.labels(weight)) == set(UNION_LABELS) | m56_minted(weight, "hf")

    reported = graphed.points(weight)
    for tag in TAGS:
        for direction in TAGS:
            joint = f"hf_{tag}__jes_{direction}"
            assert reported[joint] == {"hf": tag, "jes": direction}
            # the joint is its own node: neither one-at-a-time universe it crosses
            node = graphed.member_of(weight, joint).node_id
            assert node != graphed.member_of(weight, f"hf_{tag}").node_id, joint
            assert node != graphed.member_of(weight, f"jes_{direction}").node_id, joint
