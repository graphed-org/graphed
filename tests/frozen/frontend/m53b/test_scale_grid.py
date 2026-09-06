"""vary-m53b §5, plan §7.2 (tour cell 32): the μRxμF 7-point scale grid, spelled in the unified LIST
form. Four axis-aligned universes come from the independent ``muR`` / ``muF`` one-at-a-time families;
the two correlated DIAGONALS are additive off-grid points re-pointing an independent ``scale`` member
to a two-coordinate ``{muR, muF}`` point (own ``scale`` axis dropped). Numeric coordinates
canonicalize: ``2`` → ``"2"`` and ``0.5`` → ``"5em1"`` (probed on the working prune path).

Non-vacuity: the diagonal entries are refused today ("names no joint the fanout of 'scale' derives;
the derived joints are []"), so the ``vary`` call fails for feature-absence. The witness is that
``scale_dndn`` reports ``{muR: 5em1, muF: 5em1}`` — a two-coordinate off-grid point — not the default
``{scale: dndn}`` a bare mint leaves.
"""

from __future__ import annotations

from m53b_offgrid_fixtures import scale_grid

import graphed

SEVEN_POINTS = {
    "nominal": {},
    "muR_2": {"muR": "2"},
    "muR_5em1": {"muR": "5em1"},
    "muF_2": {"muF": "2"},
    "muF_5em1": {"muF": "5em1"},
    "scale_upup": {"muR": "2", "muF": "2"},  # correlated diagonal, own axis dropped
    "scale_dndn": {"muR": "5em1", "muF": "5em1"},  # correlated diagonal, own axis dropped
}


def _grid() -> dict[str, dict[str, str]]:
    _session, mu_f, pt = scale_grid()
    base = pt * 0.5  # independent scale factor
    registered = graphed.vary(
        mu_f,
        "scale",
        base,
        is_weight=True,
        points=[
            ("upup", base * 1.5),
            ("dndn", base * 0.87),
            {"scale": "upup", "muR": 2, "muF": 2},
            {"scale": "dndn", "muR": 0.5, "muF": 0.5},
        ],
    )
    return graphed.points(graphed.weight(registered))


def test_the_grid_has_exactly_seven_universes() -> None:
    points = _grid()
    assert len(points) == 7  # 1 nominal + 4 axis-aligned + 2 diagonals


def test_the_diagonals_are_two_coordinate_off_grid_points() -> None:
    points = _grid()
    for diagonal in ("scale_upup", "scale_dndn"):
        assert len(points[diagonal]) == 2, points[diagonal]  # a genuine μRxμF cross
        assert "scale" not in points[diagonal]  # own axis dropped
    assert points["scale_upup"] == {"muR": "2", "muF": "2"}
    assert points["scale_dndn"] == {"muR": "5em1", "muF": "5em1"}


def test_the_grid_matches_the_seven_point_map_and_keeps_the_labels() -> None:
    points = _grid()
    assert points == SEVEN_POINTS  # the whole registry, labels included
    for label in ("scale_upup", "scale_dndn"):
        assert label in points  # the analyst's spelling survives the re-point
