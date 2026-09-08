"""m55 review folds: the `**tags` spelling, refusal-before-mint witnessed through the session's
shift-after-weight diagnostic, the rollback of that diagnostic on a hand-form refusal, and the
two widened messages."""

from __future__ import annotations

from typing import Any

import pytest
from m55_lockstep_fixtures import (
    JER,
    JES_TAGS,
    OFF_NOMINAL,
    base_toy,
    jes_containers,
    jes_context,
    rescale,
    spelled,
)

import graphed
from graphed.awkward import gak
from graphed.errors import GraphedError


def _weighted_jes_context() -> tuple[Any, Any]:
    toy = base_toy()
    ctx = jes_context(toy)
    w = gak.sum(ctx.Jet.pt, axis=1)
    return toy, graphed.vary(ctx, "sf", w, is_weight=True, up=w * 1.02, down=w * 0.98)


def _diagnostics(session: Any) -> tuple[dict[Any, Any], list[Any]]:
    return dict(session._shift_after_weight), list(session._weight_factors)


def test_the_kwargs_spelling_takes_a_varied_member() -> None:
    toy = base_toy()
    by_kwarg = graphed.vary(toy.ctx, "jes", **jes_containers(toy.ctx))

    hand_toy = base_toy()
    hand = jes_containers(hand_toy.ctx)
    by_hand = graphed.vary(hand_toy.ctx, "jes", collections=spelled("hand", hand, "jes", JES_TAGS))
    for collection in ("Jet", "MET"):
        assert set(graphed.labels(by_kwarg[collection])) == {"nominal", "jes_up", "jes_down"}
        for label in ("nominal", "jes_up", "jes_down"):
            assert (
                graphed.member_of(by_kwarg[collection], label).node_id
                == graphed.member_of(by_hand[collection], label).node_id
            ), (collection, label)


def test_a_refusal_on_the_second_collection_precedes_the_first_collections_mint() -> None:
    toy, wctx = _weighted_jes_context()
    jets = graphed.nominal(wctx["Jet"])
    good = graphed.vary(jets, "jer", **{t: rescale(jets, f) for t, f in JER.items()})
    off = rescale(graphed.nominal(wctx["MET"]), OFF_NOMINAL)
    bad = graphed.vary(off, "jer", **{t: rescale(off, f) for t, f in JER.items()})

    before = _diagnostics(toy.session)
    with pytest.raises(GraphedError, match="MET"):
        graphed.vary(wctx, "jer", collections={"Jet": good, "MET": bad})
    assert _diagnostics(toy.session) == before


def test_a_hand_form_refusal_on_the_second_collection_rolls_the_diagnostic_back() -> None:
    toy, wctx = _weighted_jes_context()
    jets = graphed.nominal(wctx["Jet"])
    shifted = {t: rescale(jets, f) for t, f in JER.items()}

    before = _diagnostics(toy.session)
    with pytest.raises(GraphedError, match="its form"):
        graphed.vary(wctx, "jer", collections={"Jet": shifted, "MET": shifted})
    assert _diagnostics(toy.session) == before


def test_the_declaring_points_message_names_both_collection_value_shapes() -> None:
    toy = base_toy()
    with pytest.raises(
        GraphedError, match=r"collections=\{Name: \{tag: record\}\} or collections=\{Name: varied\}"
    ):
        graphed.vary(toy.ctx, "jes", collections=jes_containers(toy.ctx), points={"jes": 1.0})


def test_the_lockstep_message_names_a_varied_as_the_second_accepted_shape() -> None:
    toy = base_toy()
    not_a_shape: Any = {"Jet": 1.0}
    with pytest.raises(GraphedError, match=r"needs a \{tag: record\} mapping or a Varied, got float"):
        graphed.vary(toy.ctx, "jes", collections=not_a_shape)
