"""C4 / design §4.11-4, §8-i, §8-b and R2 verbatim: a typed coordinate reaches a real universe or
the call fails naming what IS registered.

Every `points=` map here carries at least two coordinates. A one-coordinate explicit point is
unregistrable by construction: §4.11-4 requires its coordinate to be a registered tag of its
nuisance, which means the default label for that (nuisance, tag) already owns the point, and
§4.11-2 refuses a second label for it.
"""

from __future__ import annotations

import pytest
from m52_point_fixtures import (
    CARRIER_POINTS,
    CARRIERS,
    JOINT_FACTOR,
    carrier,
    carrier_weight,
    identifier_families,
    materialized,
    numeric_families,
    plain_context,
    source,
    two_axis_loose,
)

import graphed
from graphed.errors import GraphedError

R2_LABEL = "jesbtag_corr_up"


def test_r2_verbatim_against_numerically_tagged_families() -> None:
    """R2 as the request spells it: a universe named by a sparse `{nuisance: coordinate}` map of
    NUMBERS, with an explicitly supplied value."""
    session, ctx = numeric_families()
    ones = ctx["pt"] * 0.0 + 1.0

    registered = graphed.vary(
        ctx,
        "jesbtag_corr",
        ones,
        is_weight=True,
        variations={"up": ones * JOINT_FACTOR},
        points={"up": {"jes": 1, "btag": -1}},
    )

    assert R2_LABEL in graphed.labels(registered)
    assert graphed.points(registered)[R2_LABEL] == {"btag": "m1", "jes": "1"}

    universes = materialized(session, graphed.weight(registered))
    joint = universes.pop(R2_LABEL)
    assert universes  # the comparison below would otherwise be vacuous
    for label, values in universes.items():
        assert joint != values, f"the joint universe collapsed onto {label}"


def test_a_typed_coordinate_naming_no_registered_tag_is_refused_naming_what_is() -> None:
    """§8-i: the refusal fires on the analyst's own spelling — `1` typed against a family the
    analyst registered `up` / `down`. Without it R2's own example returns nominal kinematics."""
    _s1, admitted = identifier_families()
    admitted_weight = admitted["pt"] * 0.5
    registered = graphed.vary(
        admitted,
        "corr",
        admitted_weight,
        is_weight=True,
        variations={"a": admitted_weight * 3.0},
        points={"a": {"jes": "up", "btag": "up"}},
    )
    assert graphed.points(registered)["corr_a"] == {"btag": "up", "jes": "up"}

    _s2, ctx = identifier_families()
    weight = ctx["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"a": weight * 3.0},
            points={"a": {"jes": 1, "btag": "up"}},
        )

    message = str(caught.value)
    assert "jes" in message
    assert "up" in message
    assert "down" in message


def test_a_nuisance_registered_nowhere_is_refused() -> None:
    _s1, admitted = identifier_families()
    admitted_weight = admitted["pt"] * 0.5
    assert graphed.labels(
        graphed.vary(
            admitted,
            "corr",
            admitted_weight,
            is_weight=True,
            variations={"a": admitted_weight * 3.0},
            points={"a": {"jes": "up", "btag": "up"}},
        )
    )

    _s2, ctx = identifier_families()
    weight = ctx["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"a": weight * 3.0},
            points={"a": {"nosuch": "up", "jes": "up"}},
        )

    message = str(caught.value)
    assert "nosuch" in message
    assert "jes" in message


def test_a_joint_point_registered_before_its_axis_exists_is_refused() -> None:
    """Without §4.11-4 the analyst gets a b-tag-only universe wearing a joint name."""
    _s1, early = plain_context()
    early_weight = early["pt"] * 0.5
    tagged = graphed.vary(
        early,
        "btag",
        early_weight,
        is_weight=True,
        variations={"up": early_weight * 1.2, "down": early_weight * 0.8},
    )
    ambient = graphed.weight(tagged)
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            tagged,
            "corr",
            ambient,
            is_weight=True,
            variations={"a": ambient * 3.0},
            points={"a": {"jes": "up", "btag": "up"}},
        )
    assert "jes" in str(caught.value)

    # reordered: the same registration with the `jes` axis already in place
    _s2, ctx = plain_context()
    pt = ctx["pt"]
    shifted = graphed.vary(ctx, "jes", collections={"pt": {"up": pt * 1.1, "down": pt * 0.9}})
    later_weight = shifted["pt"] * 0.5
    tagged_later = graphed.vary(
        shifted,
        "btag",
        later_weight,
        is_weight=True,
        variations={"up": later_weight * 1.2, "down": later_weight * 0.8},
    )
    later_ambient = graphed.weight(tagged_later)
    registered = graphed.vary(
        tagged_later,
        "corr",
        later_ambient,
        is_weight=True,
        variations={"a": later_ambient * 3.0},
        points={"a": {"jes": "up", "btag": "up"}},
    )
    assert graphed.points(registered)["corr_a"] == {"btag": "up", "jes": "up"}


@pytest.mark.parametrize("name", CARRIERS)
def test_the_carrier_walk_covers_all_three_context_carriers(name: str) -> None:
    """§4.11-4's carrier list — the ambient weight, the `Varied` collections and the selection. Each
    context here supplies its point's axes through ONE of the three and nothing else; a walk that
    reads the carriers' `_tags` rather than the registry's points over their labels refuses the
    ambient-weight case, whose weight has an EMPTY tag map."""
    _session, ctx = carrier(name)
    point = CARRIER_POINTS[name]
    weight = carrier_weight(ctx)

    registered = graphed.vary(
        ctx, "corr", weight, is_weight=True, variations={"a": weight * 3.0}, points={"a": point}
    )
    assert graphed.points(registered)["corr_a"] == point


def test_a_nuisance_on_none_of_the_three_carriers_is_refused() -> None:
    _session, ctx = carrier("ambient_only_context")
    weight = carrier_weight(ctx)

    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"a": weight * 3.0},
            points={"a": {"nowhere": "up", "jes": "up"}},
        )
    assert "nowhere" in str(caught.value)


def test_the_loose_forms_carrier_is_the_targets_own_tag_map() -> None:
    """The loose form's carrier list is `target._tags` alone."""
    _session, target = two_axis_loose()
    value = graphed.nominal(target) * 3.0

    registered = graphed.vary(
        target, "corr", variations={"a": value}, points={"a": {"jes": "up", "jer": "up"}}
    )
    assert graphed.points(registered)["corr_a"] == {"jer": "up", "jes": "up"}

    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            target, "corr2", variations={"a": value}, points={"a": {"nosuch": "up", "jes": "up"}}
        )
    assert "nosuch" in str(caught.value)


def test_an_inherited_label_with_partial_coverage_still_falls_back_silently() -> None:
    """§4.11-4's last clause and §4.7: partial coverage across two containers of ONE family is a
    legitimate pattern, and turning it into an error would break it."""
    session, record = source()
    x = record["pt"]
    left = graphed.vary(x, "jes", up=x * 10.0)
    right = graphed.vary(x, "jes", up2=x * 100.0)

    product = left * right
    assert list(graphed.labels(product)) == ["nominal", "jes_up", "jes_up2"]

    universes = materialized(session, product)
    left_universes = materialized(session, left)
    right_universes = materialized(session, right)

    def _product(one: tuple[float, ...], other: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(a * b for a, b in zip(one, other, strict=True))

    assert universes["nominal"] == _product(left_universes["nominal"], right_universes["nominal"])
    assert universes["jes_up"] == _product(left_universes["jes_up"], right_universes["nominal"])
    assert universes["jes_up2"] == _product(
        left_universes["nominal"], right_universes["jes_up2"]
    )

    # the positive control §8-b names: if the family guard is dead, the null result above is void
    with pytest.raises(GraphedError) as caught:
        graphed.vary(left, "jes", up=x * 3.0)
    assert "already registered under" in str(caught.value)
