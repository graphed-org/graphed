"""§2.5's shift-after-weight diagnostic (m49): a weight factor registered BEFORE the collection it
reads is varied fills every shift universe with its pre-shift value, and §2.1's ordering rule makes
that unfixable after the fact. Detection is RECORD-time — both operands are live only at the shift
`vary` call — and the report is a second additive `CompiledGraph` field of sorted
`(factor family, collection)` pairs, empty when the order is sound.

The fixture registers an ambient weight, so it is an event-context program (§2.1(b)).

TWO positive controls, because the two ways of reporting nothing fail differently: the correct ORDER
(the shift precedes the weight, so no ambient weight exists to test) and a correctly-ordered weight
that legitimately does not read the shifted collection (an ambient weight exists and the walk
answers no). The violating program carries TWO families over TWO shifted collections so the report
pins the PAIRING, not membership: a cross-product implementation answers with four pairs.
"""

from __future__ import annotations

from typing import Any

from m49_ctx_fixtures import events_context, jes_collections, jet_weight, met_weight, weight_universes

import graphed
from graphed import Session, compile_ir
from graphed.execute import CompiledGraph


def _compiled(session: Session, ctx: Any) -> CompiledGraph:
    return compile_ir(session, *weight_universes(ctx))


def _with_both_weights(ctx: Any) -> Any:
    """`btag` reads Jet, `pu` reads MET — disjoint cones over the two shifted collections."""
    ctx = graphed.vary(
        ctx, "btag", jet_weight(ctx), is_weight=True, up=jet_weight(ctx, 1.1), down=jet_weight(ctx, 0.9)
    )
    return graphed.vary(
        ctx, "pu", met_weight(ctx), is_weight=True, up=met_weight(ctx, 1.1), down=met_weight(ctx, 0.9)
    )


def test_a_weight_registered_before_the_shift_it_reads_is_reported_with_its_collection() -> None:
    session, ctx = events_context()
    weighted = _with_both_weights(ctx)
    shifted = graphed.vary(weighted, "jes", collections=jes_collections(weighted))

    assert _compiled(session, shifted).shift_after_weight == (("btag", "Jet"), ("pu", "MET"))


def test_a_weight_registered_after_the_shift_reports_nothing() -> None:
    """Positive control one: the correct order — the shift precedes every registration."""
    session, ctx = events_context()
    shifted = graphed.vary(ctx, "jes", collections=jes_collections(ctx))
    weighted = _with_both_weights(shifted)

    assert _compiled(session, weighted).shift_after_weight == ()


def test_an_ambient_weight_that_does_not_read_the_shifted_collection_reports_nothing() -> None:
    """Positive control two: an ambient weight EXISTS and the cone walk answers no."""
    session, ctx = events_context()
    weighted = graphed.vary(
        ctx, "pu", met_weight(ctx), is_weight=True, up=met_weight(ctx, 1.1), down=met_weight(ctx, 0.9)
    )
    jes = jes_collections(weighted)
    shifted = graphed.vary(weighted, "jes", collections={"Jet": jes["Jet"]})

    assert graphed.labels(graphed.weight(shifted)) == ("nominal", "pu_up", "pu_down")
    assert _compiled(session, shifted).shift_after_weight == ()
