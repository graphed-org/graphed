"""Impl-review witnesses the frozen m48 tree cannot reach (F1, F2, F3, F5).

Each one guards a rule the plan binds and the frozen suite leaves unwitnessed: a mutation of the
named line leaves all 112 frozen tests green. The m49 freeze carries the frozen versions; these
keep the rules from regressing meanwhile.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest
from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import GraphedError, Session, compile_ir
from graphed.awkward import AwkwardBackend, from_awkward, gak

# The frozen m48 fixtures are not importable from here (their dir is deliberately off the
# pythonpath), so the few shapes these witnesses need are spelled locally.
EVENTS = make_events(n_events=200, seed=48)


def context_with_root() -> tuple[Session, Any, Any]:
    """The context AND the context-free root read it wraps, for the seams that need both."""
    session = Session(AwkwardBackend())
    root = from_awkward(session, "events", EVENTS)
    return session, ga.gnano.events(root), root


def events_context() -> tuple[Session, Any]:
    session, context, _root = context_with_root()
    return session, context


def shifted_jets(source: Any, factor: float) -> Any:
    jets = source.Jet
    return gak.with_field(jets, jets.pt * factor, "pt")


def shifted_met(source: Any, factor: float) -> Any:
    met = source.MET
    return gak.with_field(met, met.pt * factor, "pt")


def jes_kwargs(source: Any) -> dict[str, dict[str, Any]]:
    """The lockstep shift form: Jet and MET move together."""
    return {
        "Jet": {"up": shifted_jets(source, 1.05), "down": shifted_jets(source, 0.95)},
        "MET": {"up": shifted_met(source, 1.02), "down": shifted_met(source, 0.98)},
    }


def pu_weight(source: Any, scale: float) -> Any:
    return source.MET.pt * scale


def _rows(session: Any, value: Any, label: str) -> int:
    return len(ak.to_list(session.materialize(graphed.universe(value, label))))


def _deep_rows(session: Any, value: Any, label: str) -> int:
    """`label`'s rows through a NESTED container (§2.1's two-level shape), falling back to a
    level's central universe when it does not carry the label."""
    while isinstance(value, graphed.Varied):
        value = graphed.universe(value, label) if label in graphed.labels(value) else graphed.nominal(value)
    return len(ak.to_list(session.materialize(value)))


# ---- F1: §2.1's one-row-space rule for overload (a) ------------------------------------------
def test_an_ancestor_handled_target_is_re_indexed_into_the_containers_row_space() -> None:
    """The target is read at the PARENT and the new member at a mask-derived child, so the
    container's most-derived handle is the child's. Re-indexing only the new members leaves the
    nominal universe at the parent's row count — a container advertising a handle it does not
    honour, which surfaces only at execution as a broadcast error about the wrong thing.
    """
    session, events = events_context()
    sel = events[gak.num(events.Jet) >= 4]
    varied = graphed.vary(events.MET.pt, "sf", up=sel.MET.pt * 1.1)

    assert graphed.context_of(varied) is sel
    assert _rows(session, varied, "nominal") == _rows(session, varied, "sf_up")
    # the combination legal at the container's own handle must materialize, not raise
    combined = varied * sel.MET.phi
    assert _rows(session, combined, "nominal") == _rows(session, varied, "nominal")


def test_a_vary_identity_link_leaves_the_target_and_its_handle_alone() -> None:
    """The negative control. A `vary` link keeps the row space fixed, so re-indexing across it
    would re-stamp nominal's handle and lose the parent identity §2.3e pins."""
    _s, events = events_context()
    weighted = graphed.vary(events, "pu", pu_weight(events, 1.0), is_weight=True, up=pu_weight(events, 1.1))
    varied = graphed.vary(events.MET.pt, "sf", up=weighted.MET.pt * 1.1)

    assert graphed.context_of(varied) is weighted
    assert graphed.context_of(graphed.nominal(varied)) is events
    assert graphed.nominal(varied).node_id == events.MET.pt.node_id


# ---- F2: the weight form's duplicate-label check --------------------------------------------
def test_a_weight_label_colliding_with_the_ambient_registry_under_ANOTHER_name_is_refused() -> None:
    """The cross-name prefix collision: `vary(ctx, "jes_up", x=…)` and `vary(ctx, "jes", up_x=…)`
    both spell `jes_up_x`. Composing them silently would make one universe differ from nominal in
    TWO knobs, which §2.1's one-at-a-time rule forbids."""
    _s, events = events_context()
    weight = pu_weight(events, 1.0)
    first = graphed.vary(events, "jes_up", weight, is_weight=True, x=weight * 2.0)
    assert "jes_up_x" in graphed.labels(graphed.weight(first))

    with pytest.raises(GraphedError, match="already carried by this container"):
        graphed.vary(first, "jes", weight, is_weight=True, up_x=weight * 3.0)


def test_the_same_name_family_check_still_fires_and_a_fresh_label_still_registers() -> None:
    """Positive controls: the new check must not shadow the family rule, nor refuse a legal one."""
    _s, events = events_context()
    weight = pu_weight(events, 1.0)
    first = graphed.vary(events, "pu", weight, is_weight=True, up=weight * 1.1)
    with pytest.raises(GraphedError, match="already registered under"):
        graphed.vary(first, "pu", weight, is_weight=True, up=weight * 1.2)
    second = graphed.vary(first, "btag", weight, is_weight=True, up=weight * 1.3)
    assert list(graphed.labels(graphed.weight(second))) == ["nominal", "pu_up", "btag_up"]


# ---- F3: the derivation memo must separate two masks that share a nominal --------------------
def test_two_varied_masks_sharing_a_nominal_derive_DIFFERENT_contexts() -> None:
    """The memo key covers every member's node id. Keyed on the nominal alone, `ctx[mask_b]`
    silently answers with `ctx[mask_a]`'s child and every read through it lands in the wrong
    per-label row space — a wrong histogram with nothing raised."""
    session, events = events_context()

    def counted(jets: Any) -> Any:
        return gak.num(jets[jets.pt > 25.0]) >= 4  # the pt cut is what makes a shift move rows

    central = counted(events.Jet)
    mask_a = graphed.vary(central, "jes", up=counted(shifted_jets(events, 1.05)))
    mask_b = graphed.vary(central, "jes", up=counted(shifted_jets(events, 4.00)))

    nominal_ids = {graphed.nominal(mask_a).node_id, graphed.nominal(mask_b).node_id}
    assert len(nominal_ids) == 1  # the shared nominal is what makes this discriminating
    assert graphed.universe(mask_a, "jes_up").node_id != graphed.universe(mask_b, "jes_up").node_id

    sel_a, sel_b = events[mask_a], events[mask_b]
    assert sel_a is not sel_b
    assert events[mask_a] is sel_a  # canonicalisation still holds in the collapsing direction

    rows_a = _rows(session, sel_a.MET.pt, "jes_up")
    rows_b = _rows(session, sel_b.MET.pt, "jes_up")
    assert rows_a != rows_b, "the two shifts must select different row counts for this to bite"


# ---- F5: the unreached-label diagnostic on a CONTEXT-borne program ---------------------------
def test_a_shift_varied_context_whose_universes_are_all_filled_reports_nothing() -> None:
    """Labels must survive `EventContext` projection, or the diagnostic users are told to trust
    reports labels that DO reach a marked output."""
    session, events = events_context()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    outputs = [gak.sum(graphed.universe(shifted, label).MET.pt) for label in graphed.labels(shifted)]
    compiled = compile_ir(session, *outputs)
    assert compiled.unreached_labels == ()


def test_a_context_borne_label_that_reaches_no_output_IS_reported() -> None:
    """The other direction, on the same channel: registered through a context, never filled."""
    session, events = events_context()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    compiled = compile_ir(session, gak.sum(graphed.nominal(shifted).MET.pt))
    assert set(compiled.unreached_labels) == {"jes_up", "jes_down"}


# ---- A-1: the weight form's duplicate check keys on REGISTRATIONS, not the §2.4 union --------
def test_a_correlated_weight_after_a_shift_is_accepted_in_BOTH_registration_orders() -> None:
    """The ambient container's members are the §2.4 union — shift and selection labels included —
    so a check keyed on them refuses `vary(ctx, "jes", up=…)` after a jes SHIFT, which registers
    nothing under that label and stays one knob per universe."""
    _s, events = events_context()
    weight = pu_weight(events, 1.0)

    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    first = graphed.vary(shifted, "pu", weight, is_weight=True, up=weight * 1.2)
    both = graphed.vary(first, "jes", weight, is_weight=True, up=weight * 1.1)
    assert set(graphed.labels(graphed.weight(both))) >= {"nominal", "pu_up", "jes_up"}

    swapped = graphed.vary(shifted, "jes", weight, is_weight=True, up=weight * 1.1)
    swapped = graphed.vary(swapped, "pu", weight, is_weight=True, up=weight * 1.2)
    assert set(graphed.labels(graphed.weight(swapped))) >= {"nominal", "pu_up", "jes_up"}


def test_a_weight_duplicate_under_another_name_is_STILL_refused_with_a_shift_present() -> None:
    """The population-axis control: narrowing the check to registrations must not disable it. The
    shift on the context is what makes this see the axis the union-keyed version could not."""
    _s, events = events_context()
    weight = pu_weight(events, 1.0)
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    first = graphed.vary(shifted, "sf_up", weight, is_weight=True, x=weight * 2.0)
    with pytest.raises(GraphedError, match="already carried by this container"):
        graphed.vary(first, "sf", weight, is_weight=True, up_x=weight * 3.0)


def test_align_re_indexes_an_ancestor_member_across_a_VARIED_mask_link() -> None:
    """The F1 witness drives `_align` across an unvaried mask; this drives the varied-mask path,
    where each label is re-indexed by THAT label's own mask."""
    session, events = events_context()

    def counted(jets: Any) -> Any:
        return gak.num(jets[jets.pt > 25.0]) >= 4

    mask = graphed.vary(counted(events.Jet), "jes", up=counted(shifted_jets(events, 1.05)))
    sel = events[mask]
    varied = graphed.vary(events.MET.pt, "sf", up=sel.MET.pt * 1.1)

    assert graphed.context_of(varied) is sel
    # the ancestor-handled nominal is itself Varied now — re-indexed by EACH label's own mask
    assert isinstance(graphed.nominal(varied), graphed.Varied)
    for label in ("nominal", "jes_up"):
        assert _deep_rows(session, varied, label) == _deep_rows(session, sel.MET.pt, label)
    # the two labels must really select different counts, or the per-label claim is untested
    assert _deep_rows(session, varied, "nominal") != _deep_rows(session, varied, "jes_up")


# ---- A-2: the duplicate guard must fire before the row-space maps ----------------------------
def test_a_duplicate_label_shadowing_a_contexted_member_raises_the_DESIGNED_error() -> None:
    """Run after `_align`, the collision hides its member's handle from `check_members`; the
    container is then left with no handle at all and `_align` dies on an internal AttributeError
    where the duplicate-label error belongs."""
    _s, events, root = context_with_root()
    sel = events[gak.num(events.Jet) >= 4]
    # both loose members are read from the SOURCE, so they are context-FREE: once the colliding
    # label shadows the only contexted member, the container is left with no handle at all
    target = graphed.vary(root.MET.pt, "jes_up", x=sel.MET.pt * 3.0)
    assert graphed.context_of(target) is sel
    with pytest.raises(GraphedError, match="already carried by this container"):
        graphed.vary(target, "jes", up_x=root.MET.pt * 2.0)
