"""C3 / design §4.6's row-space clause: "that label's own mask" becomes "that point's own mask",
asked of the mask container by the same projection.

`accessors._follow` narrows by exact label today, so a joint universe silently selects the NOMINAL
rows. Its `project` branch already routes through `member_of`, so it becomes point-aware for free —
asserted here rather than assumed.
"""

from __future__ import annotations

import awkward as ak
from m52_projection_fixtures import (
    JOINT_LABEL,
    ONE_AT_A_TIME_LABEL,
    joint_weight_program,
    selection_program,
)

import graphed


def test_reindex_to_follows_a_label_by_its_points_own_mask() -> None:
    session, mask, selected, carrier = selection_program()

    nominal_rows = session.materialize(graphed.universe(mask, "nominal"))
    shifted_rows = session.materialize(graphed.universe(mask, "jes_up"))
    assert int(ak.sum(nominal_rows)) != int(ak.sum(shifted_rows)), (
        "the two masks select the same rows, so this test cannot see which one was applied"
    )

    reindexed = graphed.reindex_to(carrier, selected)

    for label, expected_mask in (
        (JOINT_LABEL, shifted_rows),
        (ONE_AT_A_TIME_LABEL, nominal_rows),
    ):
        parent = session.materialize(graphed.universe(carrier, label))
        narrowed = session.materialize(graphed.universe(reindexed, label))
        assert ak.to_list(narrowed) == ak.to_list(parent[expected_mask]), label


def test_the_project_branch_follows_the_same_projection() -> None:
    """`graphed.universe(ctx, L)` is the project link. Its payload is the label, and `_follow` hands
    it to `member_of`, so the joint label reaches the shifted member and the one-at-a-time label
    still reaches nominal."""
    program = joint_weight_program()
    jes_only = program.jes_only

    shifted = graphed.universe(jes_only, "jes_up")
    central = graphed.nominal(jes_only)
    assert shifted.node_id != central.node_id

    joint_projection = graphed.universe(program.context, JOINT_LABEL)
    assert graphed.reindex_to(jes_only, joint_projection).node_id == shifted.node_id

    diagonal_projection = graphed.universe(program.context, ONE_AT_A_TIME_LABEL)
    assert graphed.reindex_to(jes_only, diagonal_projection).node_id == central.node_id
