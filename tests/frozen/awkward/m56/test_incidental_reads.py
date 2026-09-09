"""m56 §2 item 4 / §4 inclusion legs: reading a factor whose member at the label is its NOMINAL node
carries nothing of that label, so the member's coordinate still fans out — exactly as the bare
`SF(varied jets)` sibling registered in the same program does.
"""

from __future__ import annotations

from m56_fanout_fixtures import m56_cut_child, m56_incidental, m56_joints, m56_minted

import graphed


def test_a_member_multiplying_the_seed_weight_still_fans_out() -> None:
    _session, ctx = m56_incidental("seed")
    weight = graphed.weight(ctx)

    assert m56_minted(weight, "bare") == m56_joints("bare", "jes")
    assert m56_minted(weight, "probe") == m56_joints("probe", "jes")


def test_a_member_multiplying_another_familys_central_still_fans_out() -> None:
    _session, ctx = m56_incidental("central")
    weight = graphed.weight(ctx)

    assert m56_minted(weight, "bare") == m56_joints("bare", "jes")
    assert m56_minted(weight, "probe") == m56_joints("probe", "jes")
    assert not m56_minted(weight, "pu")  # the unrelated family carries no coordinate of its own


def test_a_family_on_a_cut_over_the_seed_weight_fans_out_as_on_a_kinematic_cut() -> None:
    _seed_session, seed_child = m56_cut_child("seed")
    _kin_session, kinematic_child = m56_cut_child("kinematic")

    assert m56_minted(graphed.weight(seed_child), "cut") == m56_joints("cut", "jes")
    assert m56_minted(graphed.weight(kinematic_child), "cut") == m56_joints("cut", "jes")
