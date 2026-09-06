"""vary-m53b §3, §6, plan §7.4 / §7.6 — the headline: ONE ``points=`` list carrying a PRUNE entry
(over a dependent member) AND an ADDITIVE entry (over an independent member), routed per-entry by the
named member's genuine foreign dependence. The two act on DISJOINT members, so they never contend.

The ``corr`` family carries two members: ``dep`` read off the jes-varied ``pt`` (genuinely depends on
``jes`` → auto-fans over jes → PRUNE keeps the named joint), and ``ind`` read off the nominal
(independent → ADDITIVE re-points its one-at-a-time label). The own-axis rule is witnessed on the
same call: the prune label keeps its own ``corr`` axis (a genuine cross), the additive label drops it.

Non-vacuity: the additive entry is refused today ("names no joint the fanout of 'corr' derives; the
derived joints are ['corr_dep__jes_down', 'corr_dep__jes_up']" — the ``ind`` member names none of the
dep-only joints), so the mixed call fails for feature-absence.
"""

from __future__ import annotations

from m53b_offgrid_fixtures import two_axis_context

import graphed


def _mixed() -> dict[str, dict[str, str]]:
    _session, ctx = two_axis_context()  # jes on pt, jer on eta
    dependent = ctx["pt"] * 0.5  # jes-varied → the corr family fans out over jes
    independent = graphed.nominal(ctx["pt"]) * 0.7  # off the nominal → carries neither family

    registered = graphed.vary(
        ctx,
        "corr",
        dependent,
        is_weight=True,
        variations=[
            ("dep", dependent * 1.3),
            ("ind", independent * 1.1),
            {"corr": "dep", "jes": "up"},  # PRUNE: keep this joint, drop corr_dep__jes_down
            {"corr": "ind", "jes": "up", "jer": "up"},  # ADDITIVE: re-point corr_ind
        ],
    )
    return graphed.points(graphed.weight(registered))


def test_a_prune_entry_and_an_additive_entry_both_land_in_one_call() -> None:
    points = _mixed()

    # the prune entry SELECTED its joint and dropped the sibling
    assert points["corr_dep__jes_up"] == {"corr": "dep", "jes": "up"}
    assert "corr_dep__jes_down" not in points

    # the additive entry RE-POINTED the independent member
    assert points["corr_ind"] == {"jes": "up", "jer": "up"}

    # the two members are disjoint and both survive alongside the untouched one-at-a-time universes
    assert points["corr_dep"] == {"corr": "dep"}
    for base in ("nominal", "jes_up", "jes_down", "jer_up", "jer_down"):
        assert base in points, base


def test_the_own_axis_rule_holds_across_the_mixed_call() -> None:
    """§6: a prune label carries its own family axis (a genuine cross), an additive label does not."""
    points = _mixed()

    prune_point = points["corr_dep__jes_up"]
    additive_point = points["corr_ind"]

    assert "corr" in prune_point  # the machine joint IS a cross of corr with jes
    assert "corr" not in additive_point  # the re-pointed label is a naming device only
    assert set(additive_point) == {"jes", "jer"}  # foreign-only


def test_the_mixed_registry_is_deterministic_across_two_runs() -> None:
    assert repr(_mixed()) == repr(_mixed())  # a pure function of the fixed points list
