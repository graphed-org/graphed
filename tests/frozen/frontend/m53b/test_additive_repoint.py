"""vary-m53b §3.4, §5: an additive ``points=`` entry over an INDEPENDENT member RE-POINTS its
one-at-a-time label from the default ``{name: tag}`` to the foreign-only point, DROPPING the own
axis. The label is unchanged (``f"{name}_{tag}"``), no new member is minted, and the entry routes
this way on all three ``vary`` overloads (loose / weight / shift).

Non-vacuity: the off-grid entry over an independent member is refused on the current tree ("names no
joint the fanout of 'corr' derives; the derived joints are []" — probe "(b)"), so every assertion
here fails at the ``vary`` call for feature-absence. The witness is ``graphed.points()`` showing the
annotated two-axis point, NOT the default ``{corr: a}`` a bare mint would leave.
"""

from __future__ import annotations

from m53b_offgrid_fixtures import independent_weight, two_axis_context, two_axis_loose

import graphed

#: a genuinely NEW two-coordinate point over the ``jes`` + ``jer`` carriers — foreign only, no
#: ``corr`` axis; a one-coordinate point would be refused as a second name for an axis's own label
JOINT = {"jes": "up", "jer": "up"}


def test_additive_repoints_on_the_loose_overload() -> None:
    _session, target = two_axis_loose()
    value = graphed.nominal(target) * 3.0  # independent of jes / jer

    registered = graphed.vary(
        target, "corr", variations={"a": value}, points=[{"corr": "a", "jes": "up", "jer": "up"}]
    )

    point = graphed.points(registered)["corr_a"]
    assert point == JOINT  # foreign-only point
    assert "corr" not in point  # the own axis is DROPPED
    assert "corr_a" in graphed.labels(registered)  # the analyst's label is kept verbatim
    assert "corr_a__jes_up" not in graphed.labels(registered)  # no machine joint is minted


def test_additive_repoints_on_the_weight_overload() -> None:
    _session, ctx = two_axis_context()
    factor = independent_weight(ctx)  # graphed.nominal(ctx["pt"]) * 0.5 → independent

    registered = graphed.vary(
        ctx,
        "corr",
        factor,
        is_weight=True,
        variations={"a": factor * 3.0},
        points=[{"corr": "a", "jes": "up", "jer": "up"}],
    )

    point = graphed.points(graphed.weight(registered))["corr_a"]
    assert point == JOINT
    assert "corr" not in point
    assert "corr_a" in graphed.labels(graphed.weight(registered))


def test_additive_repoints_on_the_shift_overload() -> None:
    """The shift form re-points on the varied COLLECTION; an implementation that routes the keyword
    only through the weight form silently ignores a shift additive point."""
    _session, ctx = two_axis_context()
    independent = graphed.nominal(ctx["pt"]) * 3.0  # off the nominal → carries no jes / jer

    registered = graphed.vary(
        ctx, "corr", collections={"pt": {"a": independent}}, points=[{"corr": "a", "jes": "up", "jer": "up"}]
    )

    point = graphed.points(registered)["corr_a"]
    assert point == JOINT
    assert "corr" not in point
    assert "corr_a" in graphed.labels(registered)
