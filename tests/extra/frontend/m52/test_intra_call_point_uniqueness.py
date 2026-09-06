"""§4.11-2, the intra-call end.

The frozen suite's sole §4.11-2 anchor (``test_mint_refusals.test_one_point_under_two_labels_is_
refused_naming_both``) exercises only the cross-call REGISTRY end: two separate ``vary()`` calls on
two Sessions. But ``vary.py::_check_unique`` guards BOTH ends of the uniqueness class — labels
already in the Session registry AND labels minted by the SAME ``vary()`` call. Weakening its loop to
scan only ``registry.items()`` drops the intra-call half and admits two labels for one point in one
call, while the entire frozen m52 surface stays green. This anchor reds that mutant, so the second
half of the guard is witnessed.
"""

from __future__ import annotations

import os
import sys

import pytest

# the two-axis fixture lives with the frozen m52 anchors; this extra subtree runs as its own pytest
# process (scripts/run-tests.sh) without that dir on `pythonpath`, so seat it here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "frozen", "frontend", "m52"))

from m52_point_fixtures import two_axis_context

import graphed
from graphed.errors import GraphedError

#: a legal two-coordinate point over `two_axis_context`, and a second distinct one
JOINT = {"jes": "up", "jer": "up"}
OTHER_JOINT = {"jes": "down", "jer": "up"}


def test_two_labels_for_one_point_in_a_single_vary_call_are_refused() -> None:
    """One ``vary()`` minting two labels at ONE point is refused, naming both. JOINT is absent from
    the registry when the call begins, so ONLY the intra-call half of ``_check_unique`` can catch
    it — the guard the frozen registry-end anchor never reaches."""
    _session, ctx = two_axis_context()
    weight = graphed.nominal(ctx["pt"]) * 0.5  # INDEPENDENT: both placements route ADDITIVE
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            # both re-point to ONE point → corr_a and corr_b collide intra-call
            points=[
                ("a", weight * 3.0),
                ("b", weight * 4.0),
                {"corr": "a", **JOINT},
                {"corr": "b", **JOINT},
            ],
        )
    message = str(caught.value)
    assert "corr_a" in message
    assert "corr_b" in message

    # the admitted member at the SAME end: two labels at DISTINCT points in one call BOTH mint, so
    # the guard refuses only a genuine intra-call collision, not every two-label call.
    _other_session, ok = two_axis_context()
    ok_weight = graphed.nominal(ok["pt"]) * 0.5
    registered = graphed.vary(
        ok,
        "corr",
        ok_weight,
        is_weight=True,
        points=[
            ("a", ok_weight * 3.0),
            ("b", ok_weight * 4.0),
            {"corr": "a", **JOINT},
            {"corr": "b", **OTHER_JOINT},  # DISTINCT points → both mint
        ],
    )
    points = graphed.points(registered)
    assert points["corr_a"] == JOINT
    assert points["corr_b"] == OTHER_JOINT
