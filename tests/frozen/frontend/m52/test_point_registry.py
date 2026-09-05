"""C2 / design §4.4, §4.5, §4.10: every minted label carries a point, and `points=` reaches all
three `vary` overloads.

`graphed.points` and the `points=` keyword are touched only inside bodies, so this file COLLECTS
against a tree with no m52 implementation (TEST_SANITY §5-1).
"""

from __future__ import annotations

import pytest
from m52_point_fixtures import (
    DEFAULT_POINT_MAPS,
    EXISTING_SHAPES,
    build,
    shift_then_weight_context,
    two_axis_context,
    two_axis_loose,
)

import graphed

#: a legal two-coordinate point over the `two_axis_*` programs — new, so §4.11-2 cannot fire
JOINT = {"jes": "up", "jer": "up"}


@pytest.mark.parametrize("shape", EXISTING_SHAPES)
def test_every_existing_shape_reports_its_default_points(shape: str) -> None:
    """§4.5 mints a point for EVERY label, which is the premise §4.7's theorem stands on. An
    implementation that mints only on a `points=` call answers `{}` here."""
    _session, obj = build(shape)

    reported = graphed.points(obj)
    assert reported == DEFAULT_POINT_MAPS[shape]
    assert set(reported) == set(graphed.labels(obj))
    assert reported["nominal"] == {}


def test_the_ambient_weights_axes_include_the_inherited_shift_family() -> None:
    """§8-g: the ambient weight of a shift-then-weight program CARRIES `jes_up` while its `_tags`
    has no `jes` key at all. A `_tags`-derived axis set resolves every JES universe of every
    existing program to the nominal weight."""
    _session, ctx = shift_then_weight_context()
    ambient = graphed.weight(ctx)

    assert "jes_up" in graphed.labels(ambient)
    assert "jes" not in ambient._tags  # the witness that `_tags` is the wrong source

    reported = graphed.points(ambient)
    assert reported["jes_up"] == {"jes": "up"}
    assert "jes" in {nuisance for point in reported.values() for nuisance in point}


def test_points_is_accepted_on_the_loose_overload() -> None:
    _session, target = two_axis_loose()
    value = graphed.nominal(target) * 3.0

    registered = graphed.vary(target, "corr", variations={"a": value}, points={"a": JOINT})

    assert "corr_a" in graphed.labels(registered)
    assert graphed.points(registered)["corr_a"] == JOINT


def test_points_is_accepted_on_the_weight_overload() -> None:
    _session, ctx = two_axis_context()
    weight = ctx["pt"] * 0.5

    registered = graphed.vary(
        ctx, "corr", weight, is_weight=True, variations={"a": weight * 3.0}, points={"a": JOINT}
    )

    assert "corr_a" in graphed.labels(registered)
    assert graphed.points(registered)["corr_a"] == JOINT


def test_points_is_accepted_on_the_shift_overload() -> None:
    """In the shift form the `points=` keys are the COLLECTIONS' inner tags — an implementation
    that threads the keyword only through the weight form silently ignores a shift joint point."""
    _session, ctx = two_axis_context()
    pt = ctx["pt"]

    registered = graphed.vary(
        ctx, "corr", collections={"pt": {"a": pt * 3.0}}, points={"a": JOINT}
    )

    assert "corr_a" in graphed.labels(registered)
    assert graphed.points(registered)["corr_a"] == JOINT


def test_a_variation_tagged_points_still_registers_through_variations() -> None:
    """§4.4: `points` leaves BOTH keyword namespaces, exactly as `nominal` / `is_weight` /
    `variations` / `collections` already do."""
    _session, target = two_axis_loose()
    value = graphed.nominal(target) * 3.0

    registered = graphed.vary(
        target, "corr", variations={"points": value}, points={"points": JOINT}
    )

    assert "corr_points" in graphed.labels(registered)
    assert graphed.points(registered)["corr_points"] == JOINT


def test_a_collection_named_points_still_registers_through_collections() -> None:
    _session, ctx = two_axis_context()
    collection = ctx["points"]

    registered = graphed.vary(
        ctx, "corr", collections={"points": {"a": collection * 3.0}}, points={"a": JOINT}
    )

    assert "corr_a" in graphed.labels(registered)
    assert graphed.points(registered)["corr_a"] == JOINT
