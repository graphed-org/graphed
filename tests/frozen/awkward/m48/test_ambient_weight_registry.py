"""§2.6c: a derived context inherits the ambient registry with every member RE-INDEXED.

Read through `graphed.weight(ctx)` (§9.1), never through a fill — the registry is frontend state
and this anchor is about the registry, not the sink. Re-indexing is asserted ELEMENTWISE PER
LABEL against a manually re-indexed reference, never by length equality: a length-only check
passes right-length, silently mis-weighted arrays whenever per-label counts coincide, which is
exactly what "re-index every label by nominal's mask" produces.
"""

from __future__ import annotations

from typing import Any

from vary_ctx_fixtures import as_list, events_context, jes_kwargs, pu_weight

import graphed
from graphed import Session
from graphed.awkward import gak


def _program() -> tuple[Session, Any, Any, Any]:
    """Weight registered BEFORE the shift (§2.1's ordering rule), then a VARIED-mask derivation, so
    the derived registry's row set differs per label."""
    session, events = events_context()
    weighted = graphed.vary(
        events,
        "pu",
        pu_weight(events, 1.0),
        is_weight=True,
        up=pu_weight(events, 1.1),
        down=pu_weight(events, 0.9),
    )
    shifted = graphed.vary(weighted, "jes", **jes_kwargs(weighted))
    mask = gak.num(shifted.Jet[shifted.Jet.pt > 25.0]) >= 4
    return session, shifted, mask, shifted[mask]


def _member(container: Any, label: str) -> Any:
    """§2.4's fallback: a container contributes its own member for L, else its `"nominal"` one."""
    if label in graphed.labels(container):
        return graphed.universe(container, label)
    return graphed.nominal(container)


def test_a_selection_scoped_weight_leaves_the_parent_untouched() -> None:
    """The replacement for the exemplars' per-channel `deepcopy(Weights)`."""
    _s, shifted, _mask, sel = _program()
    scoped = graphed.vary(sel, "btag", pu_weight(sel, 1.2), is_weight=True, up=pu_weight(sel, 1.3))
    assert "btag_up" in graphed.labels(graphed.weight(scoped))
    assert "btag_up" not in graphed.labels(graphed.weight(sel))
    assert "btag_up" not in graphed.labels(graphed.weight(shifted))


def test_the_derived_registry_is_re_indexed_per_label_elementwise() -> None:
    session, shifted, mask, sel = _program()
    parent = graphed.weight(shifted)
    derived = graphed.weight(sel)
    assert list(graphed.labels(derived)) == ["nominal", "pu_up", "pu_down", "jes_up", "jes_down"]

    for label in graphed.labels(derived):
        reference = as_list(session.materialize(_member(parent, label)[_member(mask, label)]))
        assert as_list(session.materialize(graphed.universe(derived, label))) == reference


def test_re_indexing_every_label_by_nominals_mask_is_a_different_answer() -> None:
    """The instrument for the assertion above: if the two agreed, that test would pass under the
    implementation it exists to reject."""
    session, shifted, mask, _sel = _program()
    parent = graphed.weight(shifted)
    nominal_mask = graphed.nominal(mask)
    per_label = as_list(session.materialize(graphed.nominal(parent)[graphed.universe(mask, "jes_up")]))
    by_nominal = as_list(session.materialize(graphed.nominal(parent)[nominal_mask]))
    assert per_label != by_nominal


def test_the_ambient_weight_answers_in_the_contexts_OWN_row_space() -> None:
    """The invariant §6.4b's precondition assumes: `graphed.weight(ctx)` never answers at the
    parent's row count."""
    session, shifted, mask, sel = _program()
    parent_rows = len(as_list(session.materialize(graphed.nominal(graphed.weight(shifted)))))
    derived_rows = len(as_list(session.materialize(graphed.nominal(graphed.weight(sel)))))
    assert derived_rows < parent_rows
    assert derived_rows == sum(as_list(session.materialize(graphed.nominal(mask))))
