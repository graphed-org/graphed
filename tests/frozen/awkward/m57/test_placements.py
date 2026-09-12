"""m57 §2.5: `points=` placements keep their DECLARED member.

A placed universe is the user's value for that point: the memoised composition must not freeze it at
nominal when the placement is minted after an ambient read, and an overlay must not replace it just
because the point carries the overlay's coordinate.
"""

from __future__ import annotations

from typing import Any

from m57_dedupe_fixtures import (
    HF_TABLE,
    MU,
    PU,
    m57_ambient_values,
    m57_at,
    m57_base,
    m57_factor,
    m57_oracle_values,
    m57_overlay,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_table_members,
    m57_two_factors,
    m57_values,
    m57_weight,
)

import graphed


def test_a_placement_minted_after_an_ambient_read_keeps_its_declared_member() -> None:
    """A placement whose point names two earlier factors' coordinates resolves to its point with and
    without an intervening read: the fold onto the memoised composition must not freeze it at
    nominal."""
    results = []
    for extra_read in (False, True):
        base = m57_two_factors()
        if extra_read:
            graphed.weight(base.ctx)
        declared = m57_sf(base.jets, 0.375) * 1.0
        x_central = m57_pu(base.jets) * 0.5
        placed = graphed.vary(
            base.ctx,
            "x",
            x_central,
            is_weight=True,
            points=[("upup", declared), {"x": "upup", "pu": "up", "hf": "up"}],
        )
        weight = graphed.weight(placed)
        # the placed label resolves each other factor at its OWN coordinate in that point
        ops = [
            m57_factor("pu", base.pu, base.pu_members, x_upup=base.pu_members["up"]),
            m57_factor("hf", base.sf, base.hf_members, x_upup=base.hf_members["up"]),
            m57_factor("x", x_central, {}, x_upup=declared),
        ]
        assert graphed.points(placed)["x_upup"] == {"hf": "up", "pu": "up"}, extra_read
        assert m57_ambient_values(base.session, weight) == m57_oracle_values(
            base.session, ops, graphed.labels(weight)
        ), extra_read
        results.append(m57_ambient_values(base.session, weight)["x_upup"])

    assert results[0] == results[1]  # the read changes no value


def test_a_placement_at_a_point_carrying_the_overlays_coordinate_keeps_its_member() -> None:
    """An overlay replaces the running product at its family's labels, never at a universe another
    family PLACED at a point that carries its coordinate — at either prefix length — while the
    overlay's own universe is still its own member."""
    for prefix_length in (1, 2):
        session, ctx = m57_base()
        jets = ctx["Jet"]
        sf, pu = m57_sf(jets), m57_pu(jets)
        pu_members, hf = m57_scaled(pu, PU), m57_table_members(jets, HF_TABLE)
        after: Any = m57_weight(ctx, "pu", pu, pu_members)
        handle = graphed.weight(after) if prefix_length == 1 else None
        after = m57_weight(after, "hf", sf, hf)
        if handle is None:
            handle = graphed.weight(after)
        after = graphed.vary(after, "mu", handle, is_weight=True, points=m57_scaled(handle, MU))
        declared = m57_sf(jets, 0.1875) * 1.0
        placed = graphed.vary(
            after,
            "lf",
            sf,
            is_weight=True,
            points=[("diag", declared), {"lf": "diag", "hf": "up", "mu": "up"}],
        )
        weight = graphed.weight(placed)
        joined = m57_factor("hf", sf, hf, lf_diag=declared)
        mu_op = m57_overlay("mu", m57_scaled(handle, MU))
        ops = (
            [m57_factor("pu", pu, pu_members), mu_op, joined]
            if prefix_length == 1
            else [m57_factor("pu", pu, pu_members), joined, mu_op]
        )

        assert graphed.points(placed)["lf_diag"] == {"hf": "up", "mu": "up"}, prefix_length
        assert m57_ambient_values(session, weight) == m57_oracle_values(
            session, ops, graphed.labels(weight)
        ), prefix_length
        assert m57_values(session, graphed.member_of(weight, "mu_up")) == m57_values(
            session, m57_at(m57_scaled(handle, MU)["up"], "mu_up")
        ), prefix_length


def test_the_overlay_family_placing_its_own_universe_off_its_axis_keeps_that_member() -> None:
    """The overlay's own universes are told apart by membership in its container, not by their points:
    a universe it placed at a point made only of other families' coordinates is still its member."""
    base = m57_two_factors()
    handle = graphed.weight(base.ctx)
    members = m57_scaled(handle, MU)
    placed = graphed.vary(
        base.ctx,
        "mu",
        handle,
        is_weight=True,
        points=[*members.items(), {"mu": "down", "pu": "up", "hf": "up"}],
    )
    weight = graphed.weight(placed)

    assert graphed.points(placed)["mu_down"] == {"hf": "up", "pu": "up"}
    assert m57_values(base.session, graphed.member_of(weight, "mu_down")) == m57_values(
        base.session, m57_at(members["down"], "mu_down")
    )
    assert m57_values(base.session, graphed.member_of(weight, "mu_up")) == m57_values(
        base.session, m57_at(members["up"], "mu_up")
    )
