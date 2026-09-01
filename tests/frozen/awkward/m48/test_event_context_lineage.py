"""§2.6b: variation history is OBJECT LINEAGE, and context handles ride the frontend wrapper.

A wrapper attribute is the only sound carrier: two sibling contexts differing only in registered
weights expose collections whose reads intern to the SAME node id, so a node-id-keyed context map
cannot tell them apart, and `Provenance` is a frozen `(filename, lineno, function, source)`
dataclass with no lineage channel. Divergence errors NAME BOTH contexts — asserted here as the
`repr` of each appearing in the message, since a message naming only one sends the reader to half
the program.

The two fill-shaped halves of these clauses are `graphed-histogram`'s: a fill from the pre-`vary`
context carrying no new label, and the origination pair yielding different fill label sets. What
stays here is the frontend-observable core those assertions are a companion to.
"""

from __future__ import annotations

import pytest
from vary_ctx_fixtures import events_context, jes_kwargs, pu_weight

import graphed
from graphed import GraphedError
from graphed.awkward import gak


def _weighted(ctx: object, name: str = "pu") -> object:
    return graphed.vary(
        ctx, name, pu_weight(ctx, 1.0), is_weight=True, up=pu_weight(ctx, 1.1), down=pu_weight(ctx, 0.9)
    )


def test_vary_returns_a_new_context_and_the_input_is_unchanged() -> None:
    _s, events = events_context()
    derived = _weighted(events)
    assert derived is not events
    assert graphed.weight(events) is None  # contexts are immutable; nothing was registered here
    assert "pu_up" in graphed.labels(derived)
    assert "pu_up" not in graphed.labels(events)


def test_the_same_read_through_a_derived_context_and_its_parent_has_one_node_id_and_two_handles() -> None:
    """The ORIGINATION rule. The merge-from-inputs rule alone gets this wrong: `Session.source`
    receives no context and `record_op` merges only from `inputs`, so a derived context's reads go
    through the same root wrapper and would inherit the PARENT's handle."""
    _s, events = events_context()
    events2 = _weighted(events)
    through_parent = events.MET.pt
    through_child = events2.MET.pt
    assert through_parent.node_id == through_child.node_id  # interning is untouched (§1.2)
    assert graphed.context_of(through_parent) is events
    assert graphed.context_of(through_child) is events2


def test_ancestor_chain_inputs_unify_to_the_most_derived_context() -> None:
    _s, events = events_context()
    events2 = _weighted(events)
    sel = events2[events2.MET.pt > 20.0]
    combined = events2.MET.pt * sel.MET.pt
    assert graphed.context_of(combined) is sel


def test_divergent_contexts_raise_AT_THE_OP_naming_both() -> None:
    """§2.3e's op-level rule is EARLY detection — the fill is not the sole raiser."""
    _s, events = events_context()
    left = events[events.MET.pt > 20.0]
    right = events[events.MET.pt > 40.0]
    with pytest.raises(GraphedError) as excinfo:
        _ = left.MET.pt * right.MET.pt
    message = str(excinfo.value)
    assert repr(left) in message and repr(right) in message


def test_pure_derivations_are_canonical_so_two_reads_of_one_universe_unify() -> None:
    """A fresh-object-per-call implementation makes these siblings and fires the divergence error
    on a legal program — §2.6b memoises them on the parent instead."""
    _s, events = events_context()
    events2 = graphed.vary(events, "jes", **jes_kwargs(events))
    jets = events2.Jet[events2.Jet.pt > 25.0]
    sel = events2[gak.num(jets) >= 4]

    assert graphed.nominal(sel) is graphed.nominal(sel)
    assert graphed.universe(sel, "jes_up") is graphed.universe(sel, "jes_up")
    # a REBUILT mask interns to identical per-label node ids, which is the binding condition
    rebuilt = gak.num(events2.Jet[events2.Jet.pt > 25.0]) >= 4
    assert events2[rebuilt] is sel
    assert graphed.context_of(graphed.nominal(sel).MET.pt * graphed.nominal(sel).MET.phi) is not None


def test_divergence_is_also_caught_at_varys_OWN_construction() -> None:
    """`vary` is a combining point that is not an op, so `record_op`'s merge chokepoint never
    sees it."""
    _s, events = events_context()
    left = events[events.MET.pt > 20.0]
    right = events[events.MET.pt > 40.0]
    with pytest.raises(GraphedError) as excinfo:
        graphed.vary(left.MET.pt, "sig", up=right.MET.pt)
    message = str(excinfo.value)
    assert repr(left) in message and repr(right) in message


def test_labels_on_a_context_reports_the_shift_labels_it_carries() -> None:
    _s, events = events_context()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    assert list(graphed.labels(shifted)) == ["nominal", "jes_up", "jes_down"]


def test_universe_and_nominal_return_a_context_that_is_a_CHILD_of_the_argument() -> None:
    """So §6.1d's lineage-based unification can relate the result to its argument: a fill mixing a
    read from the child with a read from the argument unifies instead of diverging."""
    _s, events = events_context()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    for child in (graphed.universe(shifted, "jes_up"), graphed.nominal(shifted)):
        assert child is not shifted
        assert graphed.unify_contexts(shifted, child) is child
        assert graphed.unify_contexts(child, shifted) is child
