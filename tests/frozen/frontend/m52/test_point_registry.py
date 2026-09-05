"""m53 / design §2, §3: every minted label carries a point, and `points=` PRUNES the auto-fanned
grid on all three `vary` overloads.

`graphed.points` and the `points=` keyword are touched only inside bodies, so this file COLLECTS
against a tree with no m53 implementation (TEST_SANITY §5-1).
"""

from __future__ import annotations

import pytest
from m52_point_fixtures import (
    DEFAULT_POINT_MAPS,
    EXISTING_SHAPES,
    JES_DOWN,
    JES_UP,
    build,
    shift_then_weight_context,
    source,
    two_axis_context,
)

import graphed


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


def test_points_prunes_on_the_loose_overload() -> None:
    """The loose form auto-fans-out a jes-dependent member; `points=` keeps only the named joint."""
    _session, record = source()
    pt = record["pt"]
    jes = graphed.vary(pt, "jes", up=pt * JES_UP, down=pt * JES_DOWN)
    dependent = jes * 3.0  # jes-dependent, so the corr family fans out over jes

    registered = graphed.vary(
        jes, "corr", variations={"a": dependent}, points=[{"corr": "a", "jes": "up"}]
    )

    labels = graphed.labels(registered)
    assert "corr_a__jes_up" in labels  # the selected joint
    assert "corr_a__jes_down" not in labels  # its sibling, pruned
    assert "corr_a" in labels and "nominal" in labels  # base family untouched
    assert graphed.points(registered)["corr_a__jes_up"] == {"corr": "a", "jes": "up"}


def test_points_prunes_on_the_weight_overload() -> None:
    _session, ctx = two_axis_context()
    weight = ctx["pt"] * 0.5  # jes-dependent

    registered = graphed.vary(
        ctx,
        "corr",
        weight,
        is_weight=True,
        variations={"a": weight * 3.0},
        points=[{"corr": "a", "jes": "up"}],
    )

    ambient = graphed.weight(registered)
    labels = graphed.labels(ambient)
    assert "corr_a__jes_up" in labels
    assert "corr_a__jes_down" not in labels
    assert graphed.points(ambient)["corr_a__jes_up"] == {"corr": "a", "jes": "up"}


def test_points_prunes_on_the_shift_overload() -> None:
    """In the shift form the joint is minted on the varied COLLECTION; an implementation that threads
    the keyword only through the weight form silently ignores a shift joint point."""
    _session, ctx = two_axis_context()
    pt = ctx["pt"]  # jes-varied

    registered = graphed.vary(
        ctx, "corr", collections={"pt": {"a": pt * 3.0}}, points=[{"corr": "a", "jes": "up"}]
    )

    labels = graphed.labels(registered)
    assert "corr_a__jes_up" in labels
    assert "corr_a__jes_down" not in labels
    assert graphed.points(registered)["corr_a__jes_up"] == {"corr": "a", "jes": "up"}


def test_a_variation_tagged_points_still_registers_through_variations() -> None:
    """`points` leaves BOTH keyword namespaces: a variation literally tagged `points` registers
    through `variations=` while `points=` prunes the grid, exactly as before the inversion."""
    _session, record = source()
    pt = record["pt"]
    jes = graphed.vary(pt, "jes", up=pt * JES_UP, down=pt * JES_DOWN)
    dependent = jes * 3.0

    registered = graphed.vary(
        jes, "corr", variations={"points": dependent}, points=[{"corr": "points", "jes": "up"}]
    )

    labels = graphed.labels(registered)
    assert "corr_points" in labels  # the variation named `points` registered
    assert "corr_points__jes_up" in labels  # its joint kept by the prune
    assert "corr_points__jes_down" not in labels  # sibling pruned
    assert graphed.points(registered)["corr_points__jes_up"] == {"corr": "points", "jes": "up"}


def test_a_collection_named_points_still_registers_through_collections() -> None:
    _session, ctx = two_axis_context()
    collection = ctx["points"]
    pt = ctx["pt"]  # jes-varied, so the collection variation is jes-dependent

    registered = graphed.vary(
        ctx,
        "corr",
        collections={"points": {"a": collection * pt}},
        points=[{"corr": "a", "jes": "up"}],
    )

    labels = graphed.labels(registered)
    assert "corr_a__jes_up" in labels
    assert "corr_a__jes_down" not in labels
    assert graphed.points(registered)["corr_a__jes_up"] == {"corr": "a", "jes": "up"}
