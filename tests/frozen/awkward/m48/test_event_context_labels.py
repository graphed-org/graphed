"""§2.2: `graphed.labels(ctx)` is the §2.4-ordered UNION of three terms, `"nominal"` first.

(a) the ambient weight registry's labels, (b) the labels of `Varied` collections the context
CARRIES — the ones a shift-form `vary` replaced — and (c) the labels of the selection that derived
it. Three programs, each isolating a different term, because a program where one term's labels are
covered by another cannot tell an implementation that computes only that other term.

The fill-label-SUPERSET half of this clause is not here: asserting a fill's label set needs
`graphed_histogram.Histogram.fill`, so it lives in `graphed-histogram`'s flat `tests/frozen/m48`.
"""

from __future__ import annotations

from vary_ctx_fixtures import events_context, jes_kwargs, pu_weight, shifted_jets

import graphed
from graphed.awkward import gak


def test_term_a_and_term_c_together_the_union_is_not_the_masks_labels_alone() -> None:
    _s, events = events_context()
    weighted = graphed.vary(
        events,
        "pu",
        pu_weight(events, 1.0),
        is_weight=True,
        up=pu_weight(events, 1.1),
        down=pu_weight(events, 0.9),
    )
    shifted = graphed.vary(weighted, "jes", **jes_kwargs(weighted))
    sel = shifted[gak.num(shifted.Jet[shifted.Jet.pt > 25.0]) >= 4]
    assert list(graphed.labels(sel)) == ["nominal", "pu_up", "pu_down", "jes_up", "jes_down"]


def test_term_b_alone_a_shift_varied_collection_with_an_UNVARIED_derivation_mask() -> None:
    """The collection is the only label source: nothing is registered and the mask carries no
    labels, so an implementation reading only the mask answers with nothing at all."""
    _s, events = events_context()
    shifted = graphed.vary(events, "jes", Jet=jes_kwargs(events)["Jet"])
    unvaried_mask = shifted.MET.pt > 20.0
    assert not isinstance(unvaried_mask, graphed.Varied)
    sel = shifted[unvaried_mask]
    assert graphed.weight(sel) is None  # term (a) is empty
    assert list(graphed.labels(sel)) == ["nominal", "jes_up", "jes_down"]


def test_term_c_alone_a_mask_varied_through_the_LOOSE_primitive() -> None:
    """Terms (a) and (b) are both EMPTY here — no weight is registered and no shift-form `vary`
    touched the context — so this is the only program that isolates the third term."""
    _s, events = events_context()
    mask = (
        gak.num(
            graphed.vary(
                events.Jet, "jes", up=shifted_jets(events, 1.05), down=shifted_jets(events, 0.95)
            )
        )
        >= 4
    )
    sel = events[mask]
    assert graphed.weight(events) is None and graphed.weight(sel) is None
    assert "jes_up" not in graphed.labels(events)  # the context carries no varied collection
    assert list(graphed.labels(sel)) == ["nominal", "jes_up", "jes_down"]
