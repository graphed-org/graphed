"""vary-m53b §4, plan §7.5: the additive validation refusals, each authored as a CLASS — a passing
two-coordinate additive member paired with the refused single / empty / unreachable / duplicate one,
so a blanket "refuse every additive entry" implementation fails a passing member and a "mint any
entry" one fails a refused member.

Non-vacuity: except the dependent-member guardrail (a pure prune path, live today), every passing
member is an additive re-point refused on the current tree for feature-absence ("names no joint the
fanout of 'corr' derives; the derived joints are []"), so each test fails at its first ``vary`` call.
"""

from __future__ import annotations

import pytest
from m53b_offgrid_fixtures import independent_weight, triple_numeric, two_axis_context

import graphed
from graphed.errors import GraphedError

JOINT = {"jes": "up", "jer": "up"}


def _repoint(entry: dict[str, object]) -> dict[str, dict[str, str]]:
    """Register an INDEPENDENT ``corr`` member over the jes + jer carriers with one ``points=``
    entry, returning the reported registry."""
    _session, ctx = two_axis_context()
    factor = independent_weight(ctx)
    registered = graphed.vary(
        ctx, "corr", factor, is_weight=True, variations=[("a", factor * 3.0), entry]
    )
    return graphed.points(graphed.weight(registered))


def _refused(entry: dict[str, object]) -> GraphedError:
    _session, ctx = two_axis_context()
    factor = independent_weight(ctx)
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx, "corr", factor, is_weight=True, variations=[("a", factor * 3.0), entry]
        )
    return caught.value


def test_an_own_tag_naming_no_member_of_this_call_is_refused() -> None:
    assert _repoint({"corr": "a", "jes": "up", "jer": "up"})["corr_a"] == JOINT  # admitted
    assert "typo" in str(_refused({"corr": "typo", "jes": "up", "jer": "up"}))  # a typo member


def test_an_entry_with_only_its_own_coordinate_is_refused() -> None:
    assert _repoint({"corr": "a", "jes": "up", "jer": "up"})["corr_a"] == JOINT  # admitted
    # {corr: a} adds nothing to re-point onto — corr_a already is the one-at-a-time default
    _refused({"corr": "a"})


def test_a_single_foreign_coordinate_already_owned_by_its_axis_label_is_refused() -> None:
    """§4 / tour Level 20: a label is earned only for a genuinely new ≥2-coordinate point. A
    single-foreign ``{jes: up}`` is already the ``jes_up`` label's point — a second name for it."""
    assert _repoint({"corr": "a", "jes": "up", "jer": "up"})["corr_a"] == JOINT  # admitted (2-coord)
    _refused({"corr": "a", "jes": "up"})  # refused (1-coord, already owned)


def test_an_unreachable_additive_coordinate_is_refused_naming_what_is_registered() -> None:
    assert _repoint({"corr": "a", "jes": "up", "jer": "up"})["corr_a"] == JOINT  # admitted
    message = str(_refused({"corr": "a", "jes": "up", "jer": "sideways"}))
    assert "sideways" in message
    assert "jer" in message  # the axis whose registered tags it fails to name


def test_two_additive_entries_re_pointing_to_one_point_are_refused() -> None:
    """A point-collision class: two labels for one universe means two datacard bins for one thing."""
    assert _repoint({"corr": "a", "jes": "up", "jer": "up"})["corr_a"] == JOINT  # admitted (single)

    _session, ctx = two_axis_context()
    factor = independent_weight(ctx)
    with pytest.raises(GraphedError):
        graphed.vary(
            ctx,
            "corr",
            factor,
            is_weight=True,
            variations=[
                ("a", factor * 3.0),
                ("b", factor * 4.0),
                {"corr": "a", "jes": "up", "jer": "up"},
                {"corr": "b", "jes": "up", "jer": "up"},  # same foreign point as corr_a
            ],
        )


def test_an_all_zero_point_drops_to_empty_and_is_refused_but_a_zero_beside_live_survives() -> None:
    """§4's zero asymmetry: an explicit coordinate at 0 says "central" and DROPS. All-zero drops to
    empty (refused); a zero BESIDE two live coordinates drops coordinate-wise, leaving a legal
    two-coordinate point (the adversarial member the class must still ADMIT)."""

    def build(entry: dict[str, object]) -> dict[str, dict[str, str]]:
        _session, ctx, pt = triple_numeric()  # jes / btag / jer numeric families, tags 1 / -1
        base = pt * 0.5  # independent
        registered = graphed.vary(
            ctx, "corr", base, is_weight=True, variations=[("a", base * 1.1), entry]
        )
        return graphed.points(graphed.weight(registered))

    # admitted: a genuine two-coordinate off-diagonal
    assert build({"corr": "a", "jes": 1, "btag": -1})["corr_a"] == {"jes": "1", "btag": "m1"}
    # admitted adversarial: jer:0 drops, the reachable two-coordinate point survives
    assert build({"corr": "a", "jes": 1, "btag": -1, "jer": 0})["corr_a"] == {"jes": "1", "btag": "m1"}

    # refused: every foreign coordinate at 0 → empty → the central universe nominal already is
    _s, ctx, pt = triple_numeric()
    base = pt * 0.5
    with pytest.raises(GraphedError):
        graphed.vary(
            ctx, "corr", base, is_weight=True,
            variations=[("a", base * 1.1), {"corr": "a", "jes": 0, "btag": 0}],
        )


def test_a_dependent_members_uncarried_axis_is_refused_by_prune_not_silently_additive() -> None:
    """The prune guardrail (a live path today, so this test PASSES on the current tree): a DEPENDENT
    member's entry naming a foreign axis it does not carry is refused by the prune path, never
    re-routed to additive to mint a bogus universe."""
    _s1, ctx = two_axis_context()
    dependent = ctx["pt"] * 0.5  # jes-varied → the corr family fans out over jes
    kept = graphed.vary(
        ctx, "corr", dependent, is_weight=True,
        variations=[("dep", dependent * 1.3), {"corr": "dep", "jes": "up"}],
    )
    assert graphed.points(graphed.weight(kept))["corr_dep__jes_up"] == {"corr": "dep", "jes": "up"}

    _s2, ctx2 = two_axis_context()
    dep2 = ctx2["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx2, "corr", dep2, is_weight=True,
            variations=[("dep", dep2 * 1.3), {"corr": "dep", "nosuch": "up"}],
        )
    assert "nosuch" in str(caught.value)
