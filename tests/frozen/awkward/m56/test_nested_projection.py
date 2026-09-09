"""m56 §2 item 1: the factor's target at a label is the two-level, point-restricted read — the up
member's OWN universe for that label, not every universe the up member carries. A member reading the
up member projected into a DIFFERENT universe carries nothing of this label.
"""

from __future__ import annotations

from m56_fanout_fixtures import (
    HF_SF,
    JES_SF,
    TAGS,
    m56_base,
    m56_both_kind,
    m56_joints,
    m56_minted,
    m56_sf,
)

import graphed


def test_the_up_members_other_universe_is_not_a_target_at_this_label() -> None:
    _session, ctx, _seed = m56_base()
    registered, sjets = m56_both_kind(ctx)

    up_member = m56_sf(sjets, JES_SF["up"])  # the jes weight factor's `up` member, itself a container
    cross = graphed.member_of(up_member, "jes_down")
    assert cross.node_id != graphed.member_of(up_member, "jes_up").node_id  # the target at jes_up

    context = graphed.vary(
        registered,
        "probe",
        m56_sf(sjets, HF_SF["nominal"]),
        is_weight=True,
        points={tag: m56_sf(sjets, HF_SF[tag]) + 0.0 * cross for tag in TAGS},
    )
    assert m56_minted(graphed.weight(context), "probe") == m56_joints("probe", "jes")
