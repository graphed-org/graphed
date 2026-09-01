"""§6.1d's two cross-package lineage seams, anchored where each is observable.

`Histogram.fill` lives in a DIFFERENT distribution and the lineage relation between two handles is
not reachable through §9.1's other m48 accessors, so without these two verbs the fill could only
reach `graphed`'s private context object — the very thing `context_of` exists to prevent. Their
`graphed`-side consumers are none, which is why they need a `graphed`-side anchor at all: the
`graphed-histogram` frozen suite contributes nothing to `graphed`'s diff-coverage gate.

The link-kind half is asserted PER KIND, and each kind's RESULT LABELS are part of the assertion.
"""

from __future__ import annotations

from typing import Any

import pytest
from vary_ctx_fixtures import as_list, context_with_root, events_context, jes_kwargs, pu_weight

import graphed
from graphed import GraphedError, Session
from graphed.awkward import gak


def _varied_mask_program() -> tuple[Session, Any, Any, Any, Any]:
    session, events, root = context_with_root()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    mask = gak.num(shifted.Jet[shifted.Jet.pt > 25.0]) >= 4
    return session, shifted, mask, shifted[mask], root


def test_unify_contexts_answers_the_most_derived_handle_on_one_chain() -> None:
    _s, shifted, _mask, sel, _root = _varied_mask_program()
    assert graphed.unify_contexts(shifted, sel) is sel
    assert graphed.unify_contexts(sel, shifted) is sel


def test_unify_contexts_is_None_when_every_argument_is_context_free() -> None:
    assert graphed.unify_contexts() is None
    assert graphed.unify_contexts(None, None) is None


def test_unify_contexts_ignores_context_free_arguments_beside_contexted_ones() -> None:
    """§6.1d's adopt rule: a loose value alongside contexted ones does not veto the unification."""
    _s, shifted, _mask, sel, _root = _varied_mask_program()
    assert graphed.unify_contexts(None, shifted, None, sel) is sel


def test_unify_contexts_raises_the_divergence_error_naming_both() -> None:
    _s, events = events_context()
    left = events[events.MET.pt > 20.0]
    right = events[events.MET.pt > 40.0]
    with pytest.raises(GraphedError) as excinfo:
        graphed.unify_contexts(left, right)
    message = str(excinfo.value)
    assert repr(left) in message and repr(right) in message


def test_reindex_to_is_the_identity_for_a_value_already_at_the_target_or_context_free() -> None:
    _s, shifted, _mask, sel, root = _varied_mask_program()
    at_target = sel.MET.pt
    assert graphed.reindex_to(at_target, sel).node_id == at_target.node_id
    context_free = root.MET.pt  # read from the SOURCE, not through the context
    assert graphed.context_of(context_free) is None
    assert graphed.reindex_to(context_free, shifted).node_id == context_free.node_id


def test_reindex_to_raises_when_the_values_handle_is_a_DESCENDANT_or_divergent() -> None:
    """The §2.1(b) direction rule: a mask has no inverse, so nothing carries a selection-scoped
    value back up to the parent's row space."""
    _s, shifted, _mask, sel, _root = _varied_mask_program()
    with pytest.raises(GraphedError, match="descendant"):
        graphed.reindex_to(sel.MET.pt, shifted)

    _s2, events = events_context()
    other = events[events.MET.pt > 20.0]
    with pytest.raises(GraphedError):
        graphed.reindex_to(other.MET.pt, sel)


def test_link_kind_1_a_mask_derivation_makes_an_UNVARIED_value_varied() -> None:
    """Each member is re-indexed by THAT label's own mask; an implementation that unifies handles
    but never re-indexes gets the row count wrong for every non-nominal label."""
    session, shifted, mask, sel, _root = _varied_mask_program()
    value = shifted.MET.pt
    assert not isinstance(value, graphed.Varied)
    result = graphed.reindex_to(value, sel)
    assert isinstance(result, graphed.Varied)
    assert list(graphed.labels(result)) == list(graphed.labels(mask))
    for label in graphed.labels(mask):
        reference = as_list(session.materialize(value[graphed.universe(mask, label)]))
        assert as_list(session.materialize(graphed.universe(result, label))) == reference


def test_link_kind_2_a_vary_link_is_the_IDENTITY_and_leaves_the_labels_alone() -> None:
    """The row space is unchanged across a `vary` link; only registrations differ."""
    _s, events = events_context()
    weighted = graphed.vary(events, "pu", pu_weight(events, 1.0), is_weight=True, up=pu_weight(events, 1.1))
    value = events.MET.pt
    assert graphed.reindex_to(value, weighted).node_id == value.node_id

    varied_value = graphed.vary(events.MET.pt, "sig", up=events.MET.pt * 1.1)
    carried = graphed.reindex_to(varied_value, weighted)
    assert list(graphed.labels(carried)) == list(graphed.labels(varied_value))


def test_link_kind_3_a_projection_link_returns_an_UNVARIED_value_with_no_labels() -> None:
    """The rule §6.1d's bare-`hist` anchor depends on: a projection link RESETS the accumulated
    label set, so the fill computes its label set AFTER the lineage step, not before."""
    session, _shifted, _mask, sel, _root = _varied_mask_program()
    child = graphed.nominal(sel)
    value = sel.Jet.pt
    assert isinstance(value, graphed.Varied)
    projected = graphed.reindex_to(value, child)
    assert not isinstance(projected, graphed.Varied)
    reference = as_list(session.materialize(graphed.nominal(value)))
    assert as_list(session.materialize(projected)) == reference
