"""C3 / design §4.6, §7.1-2, §8-j: registration resolves by the same projection, and where it
differs from `vary.central_universe` projection wins.

3.5 and 3.6 are the design's stated discriminating pair — an implementation that keeps the member's
own nominal unconditionally fails 3.5, one that prefers any non-nominal member fails 3.6. Neither
needs the `points=` keyword: both are default-point registrations, so both fail against a tree with
no m52 implementation on the measured RELATION rather than on an absent symbol.
"""

from __future__ import annotations

import pytest
from m52_projection_fixtures import (
    JER_UP,
    JES_UP,
    events_context,
    jes_shifted_context,
)

import graphed
from graphed.awkward import gak
from graphed.errors import GraphedError


def test_a_shift_shift_joint_is_auto_fanned_and_reaches_the_inner_jes_universe() -> None:
    """JES ⊗ JER. The supplied `jer` member is itself jes-`Varied`, so the shift fanout mints the
    joint `jer_up__jes_up` that reaches the inner JES universe the one-at-a-time `jer_up` restricts
    away — no `points=` and no silent drop."""
    _session, shifted = jes_shifted_context()
    jets = shifted.Jet
    supplied = gak.with_field(jets, jets.pt * JER_UP, "pt")  # itself `Varied` over `jes`

    registered = graphed.vary(shifted, "jer", collections={"Jet": {"up": supplied}})

    cross = graphed.member_of(supplied, "jes_up")
    central = graphed.nominal(supplied)
    assert cross.node_id != central.node_id  # the two candidate answers are distinct nodes

    joint = graphed.universe(registered.Jet, "jer_up__jes_up")
    assert joint.node_id == cross.node_id
    assert joint.node_id != central.node_id

    # the one-at-a-time jer_up stays the diagonal: its point restricts the jes axis to nominal
    assert graphed.universe(registered.Jet, "jer_up").node_id == central.node_id


def test_a_default_point_registration_keeps_the_members_own_label_when_it_carries_it() -> None:
    """§8-j. The outer call is renaming a universe the member already carries; the reduction answer
    assembles a `jes_up` universe out of the member's `jes`-NOMINAL template, physically
    inconsistent in exactly the way R1-c exists to fix."""
    _session, ctx = events_context()
    x = ctx.MET.pt

    other = graphed.vary(x, "jes", up=x * 9.0)
    z = graphed.vary(x * 2.0, "jes", up=other)

    own = graphed.member_of(other, "jes_up")
    central = graphed.nominal(other)
    assert own.node_id != central.node_id

    assert graphed.universe(z, "jes_up").node_id == own.node_id
    assert graphed.universe(z, "jes_up").node_id != central.node_id


def test_a_default_point_registration_reduces_same_family_but_fans_out_a_foreign_one() -> None:
    """The same-family boundary half is unchanged: a member holding a DIFFERENT tag of that family
    still restricts to nominal, not "prefer the newest label". The cross-family half is what m53
    changes — a foreign-varied member no longer silently drops its foreign coordinate; the
    one-at-a-time label stays the diagonal while the joint is minted alongside it."""
    _session, ctx = events_context()
    x = ctx.MET.pt

    # same family: a different jes tag on the member still restricts to nominal
    other_tag = graphed.vary(x, "jes", down=x * 7.0)
    z_tag = graphed.vary(x * 3.0, "jes", up=other_tag)
    assert graphed.universe(z_tag, "jes_up") is graphed.nominal(other_tag)

    # cross family: the member carries a foreign `jer` universe
    other_family = graphed.vary(x, "jer", up=x * 5.0)
    z_family = graphed.vary(x * 4.0, "jes", up=other_family)
    cross = graphed.member_of(other_family, "jer_up")
    assert cross.node_id != graphed.nominal(other_family).node_id

    # the one-at-a-time jes_up still restricts the foreign jer axis to nominal (the diagonal)
    assert graphed.universe(z_family, "jes_up") is graphed.nominal(other_family)
    # but m53 mints the joint alongside it, reaching the real foreign universe — not a silent drop
    joint = graphed.member_of(z_family, "jes_up__jer_up")
    assert joint.node_id == cross.node_id
    assert joint.node_id != graphed.nominal(other_family).node_id

    # §8-j's own positive control: the family guard is live, so the target-vs-member asymmetry that
    # makes the case above reachable at all is real
    with pytest.raises(GraphedError) as caught:
        graphed.vary(other_tag, "jes", down=x * JES_UP)
    assert "already registered under" in str(caught.value)
