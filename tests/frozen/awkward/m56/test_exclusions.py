"""m56 §2 item 4 / §4 exclusions: what composition still is. A member that reads a registered
factor's VARIED member at a label carries that label through the composition the weight form already
performed, so it does not fan out; a spectator coordinate still collapses. Each leg's live instrument
is a sibling registration in the same program that DOES mint.
"""

from __future__ import annotations

from m56_fanout_fixtures import (
    m56_joints,
    m56_masked_child,
    m56_minted,
    m56_pure_weight,
    m56_sf,
    m56_spectator,
    m56_two_both_kind,
    m56_weight_family,
)

import graphed


def test_a_member_computed_from_the_ambient_mints_nothing_under_two_both_kind_families() -> None:
    _session, ctx, sjets = m56_two_both_kind()
    ambient = graphed.weight(ctx)

    assert not m56_minted(graphed.weight(m56_weight_family(ctx, "amb", ambient)), "amb")
    sibling = graphed.weight(m56_weight_family(ctx, "bare", m56_sf(sjets, 1.05)))
    assert m56_minted(sibling, "bare") == m56_joints("bare", "jes") | m56_joints("bare", "jer")
    # the ambient itself carries joint labels, so the exclusion is judged at those too
    assert m56_minted(ambient, "jer") == m56_joints("jer", "jes")


def test_a_member_reading_both_the_shifted_objects_and_the_ambient_mints_nothing() -> None:
    _session, ctx, sjets = m56_two_both_kind()
    mixed = m56_sf(sjets, 1.1) * graphed.weight(ctx)

    assert not m56_minted(graphed.weight(m56_weight_family(ctx, "mix", mixed)), "mix")
    sibling = graphed.weight(m56_weight_family(ctx, "bare", m56_sf(sjets, 1.05)))
    assert m56_minted(sibling, "bare") == m56_joints("bare", "jes") | m56_joints("bare", "jer")


def test_a_pure_weight_coordinate_is_composed_while_the_carried_shift_still_fans_out() -> None:
    _session, ctx, sjets = m56_pure_weight()
    member = m56_sf(sjets, 1.1) * graphed.weight(ctx)

    weight = graphed.weight(m56_weight_family(ctx, "probe", member))
    assert m56_minted(weight, "probe") == m56_joints("probe", "jes")


def test_a_member_built_at_the_parent_stays_composed_on_the_masked_child() -> None:
    _session, child, parent_ambient = m56_masked_child()

    inherited = graphed.weight(m56_weight_family(child, "inherited", parent_ambient))
    assert not m56_minted(inherited, "inherited")
    sibling = graphed.weight(m56_weight_family(child, "bare", m56_sf(child["Jet"], 1.05)))
    assert m56_minted(sibling, "bare") == m56_joints("bare", "jes")


def test_a_spectator_coordinate_collapses_while_the_carried_shift_fans_out() -> None:
    _session, ctx, sjets, inner = m56_spectator()

    weight = graphed.weight(m56_weight_family(ctx, "spec", m56_sf(sjets, 1.1) * inner))
    assert m56_minted(weight, "spec") == m56_joints("spec", "jes")
    assert not [label for label in graphed.labels(weight) if "inner" in label]
