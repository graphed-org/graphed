"""§2.6a/§2.6d: what a context refuses, and what it must NOT reserve.

Branch names are analysis-controlled and open-ended, so any reserved attribute would be a latent
collision with real tree content — every graphed operation on a context is a module function
instead. The subscript split mirrors `Array.__getitem__`'s own: a string (or list of strings)
reads tree content, an `Array`/`Varied` mask DERIVES a context, and slice/int — which `Array`
accepts — are refused here, because a slice-derived context is not scoped in m48-m51.
"""

from __future__ import annotations

import pytest
from vary_ctx_fixtures import events_context, jes_kwargs, pu_weight, reserved_names_context

import graphed
from graphed import Array, GraphedError
from graphed.awkward import gak


def test_a_data_context_refuses_BOTH_vary_forms() -> None:
    """Accepting a shift on data and dropping its labels at the fill would be the §2.5 silent
    drop, so "data fills nominal-only" is structural for every context-borne registration."""
    _s, data = events_context(is_data=True)
    with pytest.raises(GraphedError):
        graphed.vary(data, "pu", pu_weight(data, 1.0), is_weight=True, up=pu_weight(data, 1.1))
    with pytest.raises(GraphedError):
        graphed.vary(data, "jes", **jes_kwargs(data))
    # SCOPED: the loose primitive stays public, so a varied Array over data content is still
    # expressible and v1 deliberately does not bind that case
    loose = graphed.vary(data.Jet.pt, "jes", up=data.Jet.pt * 1.05)
    assert isinstance(loose, graphed.Varied)


def test_lockstep_collections_must_share_one_tag_set() -> None:
    _s, events = events_context()
    kwargs = jes_kwargs(events)
    kwargs["MET"] = {"up": kwargs["MET"]["up"]}  # Jet moves up+down, MET only up
    with pytest.raises(GraphedError):
        graphed.vary(events, "jes", **kwargs)


def test_the_context_reserves_no_names() -> None:
    _s, tree = reserved_names_context()
    assert isinstance(tree.weights, Array)
    assert isinstance(tree.vary, Array)
    assert tree["vary"].node_id == tree.vary.node_id  # string getitem IS field access
    assert graphed.weight(tree) is None  # the verb is a module function; `.weights` is tree content


def test_slice_and_int_subscripts_are_refused_naming_the_supported_forms() -> None:
    _s, events = events_context()
    for subscript in (slice(0, 1000), 0):
        with pytest.raises(GraphedError, match="mask"):
            _ = events[subscript]
    # positive controls: the two supported subscript kinds still work
    assert isinstance(events["Jet"], Array)
    assert events[gak.num(events.Jet) >= 4] is not events


def test_universe_labels_and_nominal_answer_on_both_a_Varied_and_a_context() -> None:
    """Uniform introspection (§9.1): the same three verbs, the same input shapes."""
    _s, events = events_context()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    varied_pt = graphed.vary(events.MET.pt, "sig", up=events.MET.pt * 1.1)

    assert list(graphed.labels(shifted)) == ["nominal", "jes_up", "jes_down"]
    assert list(graphed.labels(varied_pt)) == ["nominal", "sig_up"]
    assert isinstance(graphed.universe(varied_pt, "sig_up"), Array)
    assert graphed.nominal(varied_pt).node_id == events.MET.pt.node_id
    # on a CONTEXT both verbs answer with a context, never with an Array
    assert not isinstance(graphed.universe(shifted, "jes_up"), Array)
    assert not isinstance(graphed.nominal(shifted), Array)
