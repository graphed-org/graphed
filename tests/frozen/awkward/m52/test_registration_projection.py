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


def test_a_shift_shift_joint_point_keeps_the_inner_jes_universe_through_registration() -> None:
    """JES ⊗ JER. `_vary_shift` reduces each supplied member to its central universe today, so a
    shift-side joint point loses the inner JES universe before any point can reach it."""
    _session, shifted = jes_shifted_context()
    jets = shifted.Jet
    supplied = gak.with_field(jets, jets.pt * JER_UP, "pt")  # itself `Varied` over `jes`

    registered = graphed.vary(
        shifted,
        "jer",
        collections={"Jet": {"up": supplied}},
        points={"up": {"jer": "up", "jes": "up"}},
    )

    member = graphed.universe(registered.Jet, "jer_up")
    cross = graphed.member_of(supplied, "jes_up")
    central = graphed.nominal(supplied)
    assert cross.node_id != central.node_id
    assert member.node_id == cross.node_id
    assert member.node_id != central.node_id


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


def test_a_default_point_registration_still_reduces_a_different_tag_or_family_to_nominal() -> None:
    """The boundary half: a member holding a DIFFERENT tag of that family, or only other families,
    still restricts to nominal. This is not "prefer the newest label"."""
    _session, ctx = events_context()
    x = ctx.MET.pt

    other_tag = graphed.vary(x, "jes", down=x * 7.0)
    z_tag = graphed.vary(x * 3.0, "jes", up=other_tag)
    assert graphed.universe(z_tag, "jes_up") is graphed.nominal(other_tag)

    other_family = graphed.vary(x, "jer", up=x * 5.0)
    z_family = graphed.vary(x * 4.0, "jes", up=other_family)
    assert graphed.universe(z_family, "jes_up") is graphed.nominal(other_family)

    # §8-j's own positive control: the family guard is live, so the target-vs-member asymmetry that
    # makes the case above reachable at all is real
    with pytest.raises(GraphedError) as caught:
        graphed.vary(other_tag, "jes", down=x * JES_UP)
    assert "already registered under" in str(caught.value)
