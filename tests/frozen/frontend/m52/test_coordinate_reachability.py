"""m53 / design §3, §4.11-4, §8-i, §8-b and R2: a typed coordinate reaches a real MEMBER universe or
the call fails naming what IS registered.

Every `points=` entry here carries at least two coordinates — the own-family tag plus one foreign
coordinate. A one-coordinate explicit point is unregistrable by construction: §4.11-4 requires its
coordinate to be a registered tag of its nuisance, which means the default label for that
(nuisance, tag) already owns the point, and §4.11-2 refuses a second label for it.
"""

from __future__ import annotations

import pytest
from m52_point_fixtures import (
    JOINT_FACTOR,
    identifier_families,
    materialized,
    numeric_families,
    plain_context,
    source,
    two_axis_context,
)

import graphed
from graphed.context import EventContext
from graphed.errors import GraphedError
from graphed.varied import rebuild


def test_r2_verbatim_against_numerically_tagged_families() -> None:
    """R2 as the request spells it: a universe named by a sparse `{nuisance: coordinate}` map of
    NUMBERS. Under m53 the numeric coordinate is a precision point pruned from the numeric fanout."""
    session, ctx = numeric_families()
    factor = ctx["pt"] * 0.5  # jes-dependent, carrying the numeric jes universes '1' / '-1'

    registered = graphed.vary(
        ctx,
        "jesbtag_corr",
        factor,
        is_weight=True,
        variations={"up": factor * JOINT_FACTOR},
        points=[{"jesbtag_corr": "up", "jes": 1}],
    )

    reported = graphed.points(graphed.weight(registered))
    joints = [
        label for label, point in reported.items() if point == {"jes": "1", "jesbtag_corr": "up"}
    ]
    assert len(joints) == 1, reported  # the numeric coordinate resolved to exactly one universe

    universes = materialized(session, graphed.weight(registered))
    joint = universes.pop(joints[0])
    assert universes  # the comparison below would otherwise be vacuous
    for label, values in universes.items():
        assert joint != values, f"the joint universe collapsed onto {label}"


def test_a_typed_coordinate_naming_no_registered_tag_is_refused_naming_what_is() -> None:
    """§8-i: the refusal fires on the analyst's own spelling — `1` typed against a family the
    analyst registered `up` / `down`. Without it R2's own example returns nominal kinematics."""
    _s1, admitted = identifier_families()
    admitted_weight = admitted["pt"] * 0.5  # jes-dependent
    registered = graphed.vary(
        admitted,
        "corr",
        admitted_weight,
        is_weight=True,
        variations={"a": admitted_weight * 3.0},
        points=[{"corr": "a", "jes": "up"}],
    )
    assert graphed.points(graphed.weight(registered))["corr_a__jes_up"] == {"corr": "a", "jes": "up"}

    _s2, ctx = identifier_families()
    weight = ctx["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"a": weight * 3.0},
            points=[{"corr": "a", "jes": 1}],
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
            points=[{"corr": "a", "jes": "up"}],
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
            points=[{"corr": "a", "nosuch": "up"}],
        )

    message = str(caught.value)
    assert "nosuch" in message
    assert "jes" in message


def test_a_joint_point_registered_before_its_axis_exists_is_refused() -> None:
    """Without §4.11-4 the analyst gets a b-tag-only universe wearing a joint name."""
    # jes is registered nowhere → the point's jes coordinate resolves to no member universe
    _s1, early = plain_context()
    early_weight = early["pt"] * 0.5  # plain_context has no jes → independent
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            early,
            "corr",
            early_weight,
            is_weight=True,
            variations={"a": early_weight * 3.0},
            points=[{"corr": "a", "jes": "up"}],
        )
    assert "jes" in str(caught.value)

    # reordered: the jes axis is in place, and the member carries it, so the point is reachable
    _s2, ctx = plain_context()
    pt = ctx["pt"]
    shifted = graphed.vary(ctx, "jes", collections={"pt": {"up": pt * 1.1, "down": pt * 0.9}})
    later_weight = shifted["pt"] * 0.5  # jes-dependent
    registered = graphed.vary(
        shifted,
        "corr",
        later_weight,
        is_weight=True,
        variations={"a": later_weight * 3.0},
        points=[{"corr": "a", "jes": "up"}],
    )
    assert graphed.points(graphed.weight(registered))["corr_a__jes_up"] == {"corr": "a", "jes": "up"}


def _ambient_carrier() -> tuple[EventContext, object]:
    """A context whose ambient WEIGHT carries `jes` while its `_tags` is EMPTY (§8-g). The corr
    member is read off that weight, so it CARRIES the jes universes a `_tags` walk cannot see."""
    session, record = source()
    pt = record["pt"]
    base = pt * 0.5
    minted = graphed.vary(base, "jes", up=base * 1.1, down=base * 0.9)
    ambient = rebuild({label: graphed.universe(minted, label) for label in graphed.labels(minted)})
    ctx = EventContext(session, pt, collections={"pt": pt}, weight=ambient)
    return ctx, graphed.weight(ctx)


def _collection_carrier() -> tuple[EventContext, object]:
    """A context whose Varied COLLECTION carries `jer`; the corr member is read off it."""
    session, record = source()
    pt = record["pt"]
    collection = graphed.vary(pt, "jer", up=pt * 1.2, down=pt * 0.8)
    ctx = EventContext(session, pt, collections={"pt": collection})
    return ctx, ctx["pt"] * 0.5


@pytest.mark.parametrize(
    ("builder", "nuisance"),
    [(_ambient_carrier, "jes"), (_collection_carrier, "jer")],
)
def test_the_reachability_walk_reads_carried_labels_not_tags(builder, nuisance: str) -> None:
    """§8-g / §4.11-4: a coordinate is reachable when the MEMBER carries that universe, even when the
    carrier's `_tags` is empty (the ambient-weight case). A `_tags`-derived walk refuses it."""
    ctx, factor = builder()
    registered = graphed.vary(
        ctx,
        "corr",
        factor,
        is_weight=True,
        variations={"a": factor * 3.0},
        points=[{"corr": "a", nuisance: "up"}],
    )
    joint_label = f"corr_a__{nuisance}_up"
    assert graphed.points(graphed.weight(registered))[joint_label] == {"corr": "a", nuisance: "up"}


def test_a_nuisance_on_none_of_the_carriers_is_refused() -> None:
    _session, ctx = two_axis_context()  # jes and jer registered
    weight = ctx["pt"] * 0.5  # jes-dependent
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"a": weight * 3.0},
            points=[{"corr": "a", "nowhere": "up"}],
        )
    assert "nowhere" in str(caught.value)


def test_the_loose_forms_carrier_is_the_targets_own_tag_map() -> None:
    """The loose form's carrier is `target._tags`: a coordinate the jes-dependent member carries is
    reachable, one no family registers is refused."""
    _session, record = source()
    pt = record["pt"]
    jes = graphed.vary(pt, "jes", up=pt * 1.1, down=pt * 0.9)
    dependent = jes * 3.0

    registered = graphed.vary(jes, "corr", variations={"a": dependent}, points=[{"corr": "a", "jes": "up"}])
    assert graphed.points(registered)["corr_a__jes_up"] == {"corr": "a", "jes": "up"}

    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            jes, "corr2", variations={"a": dependent}, points=[{"corr2": "a", "nosuch": "up"}]
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
