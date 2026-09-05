"""vary-m53 §2 ordering: the one-at-a-time labels come first, the joints follow, and the sequence is
a deterministic function of registration order.

Byte-determinism across two FRESH interpreters is the re-authored `awkward/m52`
`test_the_joint_program_is_byte_deterministic_across_two_runs`; this pins the human-readable ordering
RULE that hash cannot see — every joint after every one-at-a-time label.
"""

from __future__ import annotations

from m53_fanout_fixtures import JOINT_LABELS, fanout_weight

import graphed


def test_joints_follow_all_one_at_a_time_labels_deterministically() -> None:
    labels = list(graphed.labels(fanout_weight().weight))

    joints = [label for label in labels if "__" in label]
    assert len(joints) == len(JOINT_LABELS) == 8  # the fanout minted every joint

    first_joint = min(labels.index(label) for label in joints)
    last_plain = max(index for index, label in enumerate(labels) if "__" not in label)
    assert first_joint > last_plain  # every joint follows every one-at-a-time label

    # a second, independent build reproduces the exact order — no set iteration leaks into it
    assert list(graphed.labels(fanout_weight().weight)) == labels
