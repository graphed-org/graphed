"""§2.1 stacking: inherited labels are kept, and a weight form composes label-aligned per §2.4.

The base case targets a LOOSE `Varied` — every other m48 `vary` anchor targets an `Array` or a
context, so §2.2's pairing branch and §2.1(a)'s pass-through would otherwise ship unwitnessed. The
weight half is the corpus case the matrix turns on: a b-tag weight layered on a JES-propagated
weight, which requires the factor to be read TWO levels deep.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest
from vary_ctx_fixtures import (
    as_list,
    btag_weight,
    events_context,
    jes_kwargs,
    loose_varied,
    pu_weight,
    vector_source,
)

import graphed
from graphed import GraphedError, Session
from graphed.awkward import gak
from graphed.varied import member_of, rebuild


def _corpus_shaped_selection() -> tuple[graphed.Session, object, object, object]:
    """The §2.6 sketch, in order: ambient weight FIRST, then the lockstep shift, then the
    `Varied`-mask derivation. Registering `pu` before the shift is what gives the ambient registry
    a `jes_up` member at all (§2.6c) — without it `graphed.weight(sel)` is `None`."""
    session, events = events_context()
    events = graphed.vary(
        events,
        "pu",
        pu_weight(events, 1.0),
        is_weight=True,
        up=pu_weight(events, 1.1),
        down=pu_weight(events, 0.9),
    )
    events = graphed.vary(events, "jes", **jes_kwargs(events))
    jets = events.Jet[events.Jet.pt > 25.0]
    mask = gak.num(jets) >= 4
    return session, events, mask, events[mask]


def test_stacking_on_a_loose_varied_keeps_the_members_idiom_and_the_inherited_universes() -> None:
    _s, x = vector_source()
    base = loose_varied(x)
    supplied = graphed.vary(x * 1.1, "inner", hi=x * 9.9)  # a `Varied` new member
    stacked = graphed.vary(base, "jer", up=supplied, down=x * 0.9)

    assert type(graphed.nominal(stacked)) is type(x)  # the idiom comes from the members
    assert isinstance(stacked, graphed.Varied)
    assert list(graphed.labels(stacked)) == ["nominal", "jes_up", "jes_down", "jer_up", "jer_down"]
    for inherited in ("nominal", "jes_up", "jes_down"):
        assert graphed.universe(stacked, inherited).node_id == graphed.universe(base, inherited).node_id
    # a new label's member is the provided value's CENTRAL universe, not the value itself
    assert graphed.universe(stacked, "jer_up").node_id == (x * 1.1).node_id


def _two_levels_deep(container: object, label: str) -> object:
    """§2.1's `factor[L]`: the container's member for `label`, then — when that member is itself a
    `Varied` — its own member for `label`."""
    return member_of(member_of(container, label), label)


def _values(session: Session, value: object) -> Any:
    return as_list(session.materialize(value))


def test_a_weight_vary_composes_with_the_inherited_shift_label_two_levels_deep() -> None:
    """§2.1's `old_ambient[L] x factor[L]`, with the factor evaluated in THAT label's universe.

    The property, at EVERY label: the ambient is the product of the registered factors' members,
    each read two levels deep. Asserted by VALUE against an oracle that multiplies those members
    itself — the association is the composition's own business, the multiset of operands is not.

    The wrong result: the ONE-LEVEL reading, which takes the factor container's `"nominal"` member
    whole — the b-tag SF on UNSHIFTED jets — and so misses the `ttbar_4j1b_jes_up` reference.
    """
    session, _events, _mask, sel = _corpus_shaped_selection()
    old_ambient = graphed.weight(sel)

    sjets = sel.Jet[sel.Jet.pt > 25.0]  # the corpus SF is on the pt-CUT jets, read THROUGH `sel`
    central = btag_weight(sjets)  # itself a `Varied` over the inherited jes labels
    up, down = btag_weight(sjets, 1.1), btag_weight(sjets, 0.9)
    sel2 = graphed.vary(sel, "btag", central, is_weight=True, up=up, down=down)

    ambient = graphed.weight(sel2)
    # the two registered factors, hand-built so the oracle reads the same members the registration
    # did; interning makes these the very nodes `vary` was handed
    factor = rebuild({"nominal": central, "btag_up": up, "btag_down": down})
    for label in graphed.labels(ambient):
        expected = _two_levels_deep(old_ambient, label) * _two_levels_deep(factor, label)
        assert np.allclose(
            _values(session, graphed.universe(ambient, label)),
            _values(session, expected),
            rtol=1e-12,
            atol=0.0,
        ), f"the ambient at {label!r} is not the product of the factors' {label!r} members"

    # the ONE-LEVEL reading multiplies in the SF on unshifted jets; it is a different universe (and
    # a different row space), which is what makes the two-level assertion above non-vacuous
    one_level = _two_levels_deep(old_ambient, "jes_up") * graphed.nominal(central)
    assert graphed.universe(ambient, "jes_up").node_id != one_level.node_id
    assert "btag_up" in graphed.labels(ambient)


def test_a_factor_read_at_the_parent_is_accepted_and_re_indexed_to_the_derived_row_space() -> None:
    """§2.1(b)'s ROW-SPACE positive control. `Session.record_op` performs no length check, so
    without the re-indexing the mismatch would surface only at execution."""
    session, events, mask, sel = _corpus_shaped_selection()
    sel3 = graphed.vary(sel, "sf", pu_weight(events, 1.05), is_weight=True, up=pu_weight(events, 1.1))
    weight = graphed.weight(sel3)
    mask_labels = set(graphed.labels(mask))
    for label in graphed.labels(weight):
        chosen = label if label in mask_labels else "nominal"
        rows = int(ak.sum(session.materialize(graphed.universe(mask, chosen))))
        assert len(as_list(session.materialize(graphed.universe(weight, label)))) == rows


def test_a_factor_read_through_a_DESCENDANT_context_is_a_construction_time_error() -> None:
    """Not divergence, and not re-indexable: a mask has no inverse, so no operation carries a
    selection-scoped value back up to the parent's row space."""
    _s, events, _mask, sel = _corpus_shaped_selection()
    with pytest.raises(GraphedError, match="descendant"):
        graphed.vary(
            events,
            "btag",
            btag_weight(sel.Jet),
            is_weight=True,
            up=btag_weight(sel.Jet, 1.1),
            down=btag_weight(sel.Jet, 0.9),
        )
