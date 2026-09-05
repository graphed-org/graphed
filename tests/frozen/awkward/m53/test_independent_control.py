"""vary-m53 §1: an INDEPENDENT family — members built from NOMINAL — is NOT fanned out; the
combination stays the union. This is the bound m53 preserves, and the live-harness positive control:
it passes on the pre-m53 tree too, so a run in which it also failed would prove the harness dead
rather than the fanout absent.
"""

from __future__ import annotations

from m53_fanout_fixtures import independent_program

import graphed


def test_independent_jes_and_jer_stay_the_union_of_five() -> None:
    _session, combined = independent_program()

    labels = graphed.labels(combined)
    assert set(labels) == {"nominal", "jes_up", "jes_down", "jer_hi", "jer_lo"}
    assert len(labels) == 5
    assert not [label for label in labels if "__" in label]  # no joint minted for independent
