"""C2 / design §4.11-1, §4.11-2, §4.11-3, §4.11-5 and §4.5's transaction: the construction-time
checks that make a label a globally unique name for a point within a Session.

Each refusal ships with the admitted member that must still pass, in the same test — a blanket
"refuse every re-mint" implementation has to fail one of them.
"""

from __future__ import annotations

import pytest
from m52_point_fixtures import (
    descendant_weight,
    numeric_families,
    source,
    two_axis_context,
)

import graphed
from graphed.errors import GraphedError

#: a legal two-coordinate point over `two_axis_context` / `two_axis_loose`
JOINT = {"jes": "up", "jer": "up"}
OTHER_JOINT = {"jes": "down", "jer": "up"}


def test_one_label_under_two_points_is_refused_naming_both() -> None:
    """§4.11-1. `vary(x, "jes", btag_up=…)` and `vary(y, "jes_btag", up=…)` render ONE label out of
    two different (name, tag) pairs; today neither call raises and combination silently merges them
    into a universe differing from nominal in two knobs."""
    _session, record = source()
    x = record["pt"]
    y = record["eta"]

    # the admitted member: one label, one point, minted twice on independent containers
    first = graphed.vary(x, "jes", up=x * 1.1)
    second = graphed.vary(y, "jes", up=y * 1.1)
    assert "jes_up" in graphed.labels(first)
    assert "jes_up" in graphed.labels(second)

    graphed.vary(x, "jes", btag_up=x * 5.0)  # label jes_btag_up, point {jes: btag_up}
    with pytest.raises(GraphedError) as caught:
        graphed.vary(y, "jes_btag", up=y * 7.0)  # same label, point {jes_btag: up}

    message = str(caught.value)
    assert "jes_btag_up" in message
    for named in ("jes_btag", "btag_up", "up"):
        assert named in message


def test_one_point_under_two_labels_is_refused_naming_both() -> None:
    """§4.11-2. Two labels for one universe means two slots, two StrCategory bins and two content
    hashes, and hands a datacard writer two templates for one point."""
    _session, admitted = two_axis_context()
    weight = admitted["pt"] * 0.5
    # the admitted member: a genuinely new, two-coordinate point
    registered = graphed.vary(
        admitted,
        "btag",
        weight,
        is_weight=True,
        variations={"only": weight * 3.0},
        points={"only": JOINT},
    )
    assert graphed.points(registered)["btag_only"] == JOINT

    _other_session, ctx = two_axis_context()
    other_weight = ctx["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "btag",
            other_weight,
            is_weight=True,
            variations={"only": other_weight * 3.0},
            points={"only": {"jes": "up"}},  # already the point of `jes_up`
        )

    message = str(caught.value)
    assert "jes_up" in message
    assert "btag_only" in message


def test_a_points_key_that_is_not_a_tag_of_this_call_is_refused() -> None:
    """§4.11-5, and the canonicalizing matcher it needs. Keys are matched AFTER `canonical_tag`, so
    `"0.5"` and `"5em1"` are one key while `"0p5"` and `"0.5"` are two — a matcher comparing
    `numeric_value` instead admits the second pair, both being `Fraction(1, 2)`."""
    _s1, ctx = two_axis_context()
    weight = ctx["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"up": weight * 3.0},
            points={"down": JOINT},
        )
    message = str(caught.value)
    assert "down" in message
    assert "up" in message

    # the admitted member: two DIFFERENT spellings that canonicalize alike
    _s2, spelled = two_axis_context()
    spelled_weight = spelled["pt"] * 0.5
    registered = graphed.vary(
        spelled,
        "corr",
        spelled_weight,
        is_weight=True,
        variations={"0.5": spelled_weight * 3.0},
        points={"5em1": JOINT},
    )
    assert graphed.points(registered)["corr_5em1"] == JOINT

    # the adversarial member the class must REFUSE: `canonical_tag("0p5")` is `"0p5"`, so `"0.5"`
    # is not a tag of this call even though `numeric_value` agrees on both
    _s3, adversarial = two_axis_context()
    adversarial_weight = adversarial["pt"] * 0.5
    with pytest.raises(GraphedError) as adversarial_caught:
        graphed.vary(
            adversarial,
            "corr",
            adversarial_weight,
            is_weight=True,
            variations={"0p5": adversarial_weight * 3.0},
            points={"0.5": JOINT},
        )
    assert "0p5" in str(adversarial_caught.value)


def test_an_origin_points_entry_is_refused_while_a_zero_tag_still_mints() -> None:
    """§4.11-3 and §4.2's zero asymmetry at the API: an EXPLICIT coordinate at 0 says "central", a
    registered TAG `0` names a real universe."""
    _s1, record = source()
    x = record["pt"]
    # the admitted member: the zero TAG, never zero-dropped
    zero_tag = graphed.vary(x, "shift", **{"0": x * 2.0})
    assert graphed.points(zero_tag)["shift_0"] == {"shift": "0"}

    _s2, ctx = numeric_families()
    weight = ctx["pt"] * 0.5
    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "corr",
            weight,
            is_weight=True,
            variations={"up": weight * 3.0},
            points={"up": {"jes": 0}},
        )
    message = str(caught.value)
    assert "central" in message or "nominal" in message

    # a zero coordinate BESIDE live ones drops, leaving the two-coordinate point
    _s3, survivor_ctx = numeric_families()
    survivor_weight = survivor_ctx["pt"] * 0.5
    registered = graphed.vary(
        survivor_ctx,
        "corr",
        survivor_weight,
        is_weight=True,
        variations={"up": survivor_weight * 3.0},
        points={"up": {"jes": 1, "jer": 1, "btag": 0}},
    )
    assert graphed.points(registered)["corr_up"] == {"jer": "1", "jes": "1"}


def test_a_failed_vary_leaves_the_registry_untouched_and_the_label_reregistrable() -> None:
    """§4.5's transactional mint. Without it one failed call poisons a label for the life of the
    Session, on exactly the notebook surface these spellings are iterated on."""
    _session, ctx = two_axis_context()

    before = graphed.points(ctx)
    assert before != {}  # the equality below would otherwise be vacuous

    with pytest.raises(GraphedError) as caught:
        graphed.vary(
            ctx,
            "sf",
            descendant_weight(ctx),
            is_weight=True,
            variations={"up": descendant_weight(ctx)},
            points={"up": JOINT},
        )
    assert "descendant" in str(caught.value)
    assert graphed.points(ctx) == before

    good = ctx["pt"] * 0.5
    recovered = graphed.vary(
        ctx, "sf", good, is_weight=True, variations={"up": good * 3.0}, points={"up": OTHER_JOINT}
    )
    assert graphed.points(recovered)["sf_up"] == OTHER_JOINT
