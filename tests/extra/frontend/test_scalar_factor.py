"""`graphed.vary(1.0, name, ...)`: a flat factor multiplied into a weight."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.errors import GraphedError
from graphed.systematics import ScalarFactor

EVENTS = ak.Array({"w": [1.5, 2.5, -0.5], "sf": [0.9, 1.1, 1.0]})
UP, DOWN = 1.025, 0.975


def _events(events: ak.Array = EVENTS) -> Any:
    return from_awkward("Events", events, session=Session(AwkwardBackend()))


def _lumi() -> Any:
    return graphed.vary(1.0, "lumi", up=UP, down=DOWN)


def _values(varied: Any) -> dict[str, list[float]]:
    session = graphed.nominal(varied).session
    labels = graphed.labels(varied)
    return {
        label: ak.to_list(session.materialize(graphed.universe(varied, label))) for label in labels
    }


def test_labels_of_the_product() -> None:
    weight = _events().w * _lumi()
    assert graphed.labels(weight) == ("nominal", "lumi_up", "lumi_down")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_same_values_as_the_ones_like_form_bit_for_bit(dtype: str) -> None:
    events = ak.Array({"w": ak.values_astype(EVENTS.w, dtype), "sf": EVENTS.sf})
    ev = _events(events)
    factor = _events(events)
    one = gak.ones_like(factor.w)
    by_hand = factor.w * graphed.vary(one, "lumi", up=one * UP, down=one * DOWN)
    assert _values(ev.w * _lumi()) == _values(by_hand)


def test_either_side_and_a_chain_stay_independent() -> None:
    ev = _events()
    pileup = graphed.vary(1.0, "pileup", up=1.05, down=0.95)
    weight = pileup * (ev.w * _lumi())
    assert graphed.labels(weight) == (
        "nominal",
        "lumi_up",
        "lumi_down",
        "pileup_up",
        "pileup_down",
    )
    values = _values(weight)
    assert values["pileup_up"] == [x * 1.05 for x in ak.to_list(EVENTS.w)]
    assert values["lumi_up"] == [x * UP for x in ak.to_list(EVENTS.w)]


def test_a_factor_adds_to_what_already_varies_with_no_joint() -> None:
    ev = _events()
    sf = graphed.vary(ev.sf, "muid", up=ev.sf * 1.1, down=ev.sf * 0.9)
    weight = ev.w * sf * _lumi()
    assert set(graphed.labels(weight)) == {
        "nominal",
        "muid_up",
        "muid_down",
        "lumi_up",
        "lumi_down",
    }
    assert _values(weight)["lumi_up"] == [
        w * sf * UP for w, sf in zip(EVENTS.w.tolist(), EVENTS.sf.tolist(), strict=True)
    ]


def test_a_nominal_other_than_one_scales_every_universe_and_members_are_absolute() -> None:
    ev = _events()
    values = _values(ev.w * graphed.vary(2.0, "norm", up=3.0))
    assert values["nominal"] == [x * 2.0 for x in EVENTS.w.tolist()]
    assert values["norm_up"] == [x * 3.0 for x in EVENTS.w.tolist()]


def test_one_factor_serves_two_sessions() -> None:
    lumi = _lumi()
    assert graphed.labels(_events().w * lumi) == graphed.labels(_events().w * lumi)


def test_one_factor_serves_two_weights_of_a_session() -> None:
    ev, lumi = _events(), _lumi()
    assert graphed.labels(ev.w * lumi) == graphed.labels((ev.w + 1.0) * lumi)


def test_repr_says_what_it_was_given() -> None:
    assert repr(_lumi()) == "graphed.vary(1.0, 'lumi', up=1.025, down=0.975)"


def test_it_is_a_factor_and_not_a_container() -> None:
    assert isinstance(_lumi(), ScalarFactor)
    assert not isinstance(_lumi(), graphed.systematics.Varied)
    assert not hasattr(_lumi(), "session")


@pytest.mark.parametrize(
    "use",
    [
        lambda ev: gak.sum(_lumi()),
        lambda ev: ev.w.filter(_lumi()),
        lambda ev: ev.session.materialize(_lumi()),
        lambda ev: _lumi().pt,
    ],
)
def test_where_an_array_is_required_it_says_so(use: Callable[[Any], Any]) -> None:
    with pytest.raises(AttributeError, match="the scalar factor 'lumi' is no array"):
        use(_events())


@pytest.mark.parametrize(
    "use",
    [
        lambda: _lumi() * _lumi(),
        lambda: _lumi() * 2.0,
        lambda: _lumi() * np.ones(2),
    ],
)
def test_it_multiplies_only_arrays(use: Callable[[], Any]) -> None:
    with pytest.raises(GraphedError, match="'lumi' multiplies an Array or a Varied"):
        use()


@pytest.mark.parametrize(
    "op",
    [
        lambda x, f: x + f,
        lambda x, f: x / f,
        lambda x, f: f - x,
        lambda x, f: x == f,
        lambda x, f: x != f,
        lambda x, f: (x * f) == graphed.vary(1.0, "other", up=UP),
    ],
)
def test_only_a_product_is_defined(op: Callable[[Any, Any], Any]) -> None:
    with pytest.raises(TypeError, match="unsupported operand"):
        op(_events().w, _lumi())


def test_two_factors_compare_by_identity() -> None:
    lumi = _lumi()
    assert lumi != _lumi()
    assert {lumi: "kept"}[lumi] == "kept"


@pytest.mark.parametrize(
    "options",
    [
        {"nominal": 1.0},
        {"is_weight": True},
        {"points": {"up": 1.1}},
        {"collections": {}},
        {"composes_as_union": True},
        {"max_universes": 8},
    ],
)
def test_a_factor_takes_only_its_members(options: dict[str, Any]) -> None:
    with pytest.raises(GraphedError, match="takes only its members"):
        graphed.vary(1.0, "lumi", up=UP, **options)


@pytest.mark.parametrize("members", [{}, {"up": "1.1"}, {"up": True}])
def test_a_factor_needs_number_members(members: dict[str, Any]) -> None:
    with pytest.raises(GraphedError, match="needs number members"):
        graphed.vary(1.0, "lumi", **members)


@pytest.mark.parametrize(
    ("nominal", "member"),
    [(float("nan"), UP), (float("inf"), UP), (1.0, float("nan")), (1.0, float("-inf"))],
)
def test_a_factor_needs_finite_numbers(nominal: float, member: float) -> None:
    with pytest.raises(GraphedError, match="needs finite numbers"):
        graphed.vary(nominal, "lumi", up=member)


def test_a_bool_is_no_factor() -> None:
    with pytest.raises(GraphedError, match="got bool"):
        graphed.vary(True, "lumi", up=UP)


def test_a_scalar_member_on_an_array_points_at_the_factor() -> None:
    ev = _events()
    with pytest.raises(GraphedError, match=r"got float; a flat factor is graphed\.vary\(1\.0"):
        graphed.vary(ev.w, "lumi", up=UP)


def test_the_guard_names_the_foreign_families_and_the_idiom() -> None:
    ev = _events()
    w = ev.w
    w = graphed.vary(w, "a", up=w * 1.1, down=w * 0.9)
    w = graphed.vary(w, "b", up=w * 1.1, down=w * 0.9)
    with pytest.raises(GraphedError) as caught:
        graphed.vary(w, "c", up=w * 1.1, down=w * 0.9, max_universes=8)
    message = str(caught.value)
    assert "would fan out to 27 universes (a(3) x b(3) x c(3))" in message
    assert "already vary with a, b." in message
    assert "c = graphed.vary(1.0, 'c', up=..., down=...)" in message
    assert "composes_as_union=True" in message
    assert "raise max_universes (currently 8)" in message
