"""m57 §2.3: lineage — a family registered on a mask-derived child EXTENDS the ancestor's factor.

The child adopts one composed container that stands for the adopting parent's live operations, so a
central built at the ancestor still names the factor it named there; the child's list is the parent's
live operations re-indexed with the matched one extended. Pre-m57 the adopted product is multiplied
by the ancestor's central all over again.
"""

from __future__ import annotations

from typing import Any

import pytest
from m57_dedupe_fixtures import (
    HF_TABLE,
    HT_CUT,
    JES,
    LF_TABLE,
    MET_CUT,
    MF_TABLE,
    MU,
    NF_TABLE,
    PU,
    Op,
    m57_ambient_values,
    m57_base,
    m57_delta,
    m57_factor,
    m57_joined,
    m57_oracle_values,
    m57_overlay,
    m57_pu,
    m57_reindexed,
    m57_scaled,
    m57_sf,
    m57_shifted,
    m57_table_members,
    m57_weight,
)

import graphed
from graphed import GraphedError, Varied
from graphed.awkward import gak


def _at(value: Any, ctx: Any) -> Any:
    return graphed.reindex_to(value, ctx)


def _members_at(members: Any, ctx: Any) -> dict[str, Any]:
    return {tag: graphed.reindex_to(member, ctx) for tag, member in members.items()}


def _agrees(session: Any, weight: Any, ops: list[Op], note: Any = "") -> None:
    """The ambient matches the one-SF oracle on every label the ambient carries."""
    got = m57_ambient_values(session, weight)
    assert got == m57_oracle_values(session, ops, graphed.labels(weight)), note


def _parent(*, two_factors: bool) -> tuple[Any, Any, Any, Any, list[Op]]:
    """A root context with one or two live factors over the SF central `sf`; returns the session, the
    context, the jets, that central and the expected factor operations in registration order."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    hf = m57_table_members(jets, HF_TABLE)
    ops: list[Op] = []
    after = ctx
    if two_factors:
        after = m57_weight(after, "pu", pu, m57_scaled(pu, PU))
        ops.append(m57_factor("pu", pu, m57_scaled(pu, PU)))
    after = m57_weight(after, "hf", sf, hf)
    ops.append(m57_factor("hf", sf, hf))
    return session, after, jets, sf, ops


def test_a_family_on_a_masked_child_extends_the_ancestors_factor() -> None:
    """One parent factor and two: the child's ambient is the parent's operations re-indexed with the
    matched one extended, and the parent's own ambient is untouched."""
    for two_factors in (False, True):
        session, parent, jets, sf, ops = _parent(two_factors=two_factors)
        parent_before = m57_ambient_values(session, graphed.weight(parent))
        child = parent[parent["MET"].pt > MET_CUT]
        lf = m57_table_members(jets, LF_TABLE)

        registered = m57_weight(child, "lf", _at(sf, child), _members_at(lf, child))
        weight = graphed.weight(registered)
        child_ops = [m57_reindexed(op, child) for op in ops]
        child_ops[-1] = m57_joined(child_ops[-1], "lf", _members_at(lf, child))

        _agrees(session, weight, child_ops, two_factors)
        assert m57_ambient_values(session, graphed.weight(parent)) == parent_before, two_factors


def test_the_same_central_recomputed_at_the_child_is_a_new_factor() -> None:
    """A mask makes one node id two values, so a central re-derived below it names nothing: today's
    product, which is this tree's live control."""
    session, parent, _jets, sf, ops = _parent(two_factors=True)
    child = parent[parent["MET"].pt > MET_CUT]
    re_derived = m57_sf(child["Jet"])
    lf = m57_table_members(child["Jet"], LF_TABLE)
    assert re_derived.node_id != _at(sf, child).node_id

    registered = m57_weight(child, "lf", re_derived, lf)
    child_ops = [m57_reindexed(op, child) for op in ops] + [m57_factor("lf", re_derived, lf)]

    _agrees(session, graphed.weight(registered), child_ops)


def test_a_bare_varied_over_a_live_factors_central_is_refused_as_before() -> None:
    """A central whose member at `nominal` is ITSELF a container is nested past the one level the
    weight form reads: the form check refuses it exactly as today, and it names no factor."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    central = m57_sf(sjets)
    registered = m57_weight(shifted, "hf", central, m57_table_members(sjets, HF_TABLE))
    nested = Varied({"nominal": central})

    with pytest.raises(GraphedError):
        m57_weight(registered, "probe", nested, m57_table_members(sjets, MF_TABLE))
    assert m57_ambient_values(session, graphed.weight(registered))  # the registry still answers


def test_a_crossed_handle_registers_at_the_child_beside_an_ancestor_factor_join() -> None:
    """A handle read at the parent is matched at the child as any record is — the adopted head stands
    for the ancestor's live factors — so the overlay lands behind the head in either order with a
    family that joins an ancestor's factor there."""
    for projected in (False, True):
        for delta_first in (True, False):
            session, parent, jets, sf, ops = _parent(two_factors=True)
            handle = graphed.weight(parent)
            child = graphed.nominal(parent) if projected else parent[parent["MET"].pt > MET_CUT]
            crossed = _at(handle, child)
            lf = _members_at(m57_table_members(jets, LF_TABLE), child)

            def register_mu(target: Any, crossed: Any = crossed) -> Any:
                return m57_delta(target, "mu", crossed)

            def register_lf(target: Any, child: Any = child, sf: Any = sf, lf: Any = lf) -> Any:
                return m57_weight(target, "lf", _at(sf, child), lf)

            target: Any = child
            for step in (register_mu, register_lf) if delta_first else (register_lf, register_mu):
                target = step(target)
            weight = graphed.weight(target)
            child_ops = [m57_reindexed(op, child) for op in ops]
            child_ops[-1] = m57_joined(child_ops[-1], "lf", lf)
            child_ops.append(m57_overlay("mu", m57_scaled(crossed, MU)))
            carried = {"nominal", *(label for op in child_ops for label in op.members)}
            expected = m57_oracle_values(session, child_ops, sorted(carried & set(graphed.labels(weight))))
            values = m57_ambient_values(session, weight)

            assert len(expected) > 1, (projected, delta_first)
            assert {label: values[label] for label in expected} == expected, (projected, delta_first)


def test_a_crossed_handle_over_a_strict_prefix_anchors_after_that_prefix_at_the_child() -> None:
    """A record that is a strict PREFIX of the head's tuple expands the head and anchors after its
    prefix, so a factor registered at the parent after the read still multiplies the overlay."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    hf = m57_table_members(jets, HF_TABLE)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    prefix = graphed.weight(after)
    after = m57_weight(after, "hf", sf, hf)
    child = after[met.pt > MET_CUT]
    crossed = _at(prefix, child)

    registered = m57_delta(child, "mu", crossed)
    ops = [
        m57_reindexed(m57_factor("pu", pu, m57_scaled(pu, PU)), child),
        m57_overlay("mu", m57_scaled(crossed, MU)),
        m57_reindexed(m57_factor("hf", sf, hf), child),
    ]

    _agrees(session, graphed.weight(registered), ops)


def test_a_crossed_handle_over_a_widened_factor_is_refused_as_stale() -> None:
    """A handle read over a factor a join later widened no longer names the composition it did; the
    cut taken after the join does not launder it, and the refusal names the read to repeat."""
    _session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    flat_jets = graphed.nominal(sjets)
    flat, pu = m57_sf(flat_jets), m57_pu(flat_jets)
    after = m57_weight(shifted, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", flat, m57_table_members(flat_jets, HF_TABLE))
    handle = graphed.weight(after)
    widened = m57_weight(after, "lf", m57_sf(sjets), m57_table_members(sjets, LF_TABLE))
    child = widened[widened["MET"].pt > MET_CUT]

    with pytest.raises(GraphedError) as caught:
        m57_delta(child, "mu", _at(handle, child))
    message = str(caught.value)

    assert "mu" in message and "weight(" in message
    # the live control: the handle read AFTER the join registers at the same child
    registered = m57_delta(child, "nu", _at(graphed.weight(widened), child))
    assert "nu_up" in graphed.labels(graphed.weight(registered))


def test_two_families_on_one_ancestor_central_are_one_factor_at_the_child() -> None:
    """Each re-indexed entry keeps the nominal it had in every row space it came through, so a second
    and a third family built on the ancestor's central both land on the one extended factor."""
    session, parent, jets, sf, ops = _parent(two_factors=False)
    child = parent[parent["MET"].pt > MET_CUT]
    lf = _members_at(m57_table_members(jets, LF_TABLE), child)
    mf = _members_at(m57_table_members(jets, MF_TABLE), child)

    after = m57_weight(child, "lf", _at(sf, child), lf)
    after = m57_weight(after, "mf", _at(sf, child), mf)
    expected = [m57_joined(m57_joined(m57_reindexed(ops[0], child), "lf", lf), "mf", mf)]

    _agrees(session, graphed.weight(after), expected)


def test_two_families_on_two_different_ancestor_factors_extend_each_of_them() -> None:
    """Two ancestor factors, one family naming each at the same child: the ancestor's list keeps its
    order and each entry is extended once."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    one, other = m57_sf(jets), m57_sf(jets, 1.5) * 1.0
    hf, gf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, MF_TABLE)
    after = m57_weight(ctx, "hf", one, hf)
    after = m57_weight(after, "gf", other, gf)
    child = after[after["MET"].pt > MET_CUT]
    lf = _members_at(m57_table_members(jets, LF_TABLE), child)
    nf = _members_at(m57_table_members(jets, NF_TABLE), child)

    after = m57_weight(child, "lf", _at(one, child), lf)
    after = m57_weight(after, "nf", _at(other, child), nf)
    expected = [
        m57_joined(m57_reindexed(m57_factor("hf", one, hf), child), "lf", lf),
        m57_joined(m57_reindexed(m57_factor("gf", other, gf), child), "nf", nf),
    ]

    _agrees(session, graphed.weight(after), expected)


def test_a_central_from_the_top_of_a_two_mask_chain_names_its_factor_at_every_depth() -> None:
    """Each re-indexed entry keeps the nominal it had in EVERY row space it came through, not only the
    last: the root's central still names the factor after an expansion at the child and again at the
    grandchild."""
    session, parent, jets, sf, ops = _parent(two_factors=False)
    child = parent[parent["MET"].pt > MET_CUT]
    at_child = m57_weight(child, "lf", _at(sf, child), _members_at(m57_table_members(jets, LF_TABLE), child))
    grandchild = at_child[gak.sum(at_child["Jet"].pt, axis=1) > HT_CUT]
    lf = _members_at(m57_table_members(jets, LF_TABLE), grandchild)
    mf = _members_at(m57_table_members(jets, MF_TABLE), grandchild)

    after = m57_weight(grandchild, "mf", _at(sf, grandchild), mf)
    expected = [m57_joined(m57_joined(m57_reindexed(ops[0], grandchild), "lf", lf), "mf", mf)]

    _agrees(session, graphed.weight(after), expected)


def test_an_ancestor_factor_join_after_a_crossed_overlay_agrees_in_both_orders() -> None:
    """A family joining an ancestor's factor after a strict-prefix crossed overlay joins it as it
    would before: the overlay keeps its anchored position in either registration order."""
    for delta_first in (True, False):
        session, ctx = m57_base()
        jets, met = ctx["Jet"], ctx["MET"]
        sf, pu = m57_sf(jets), m57_pu(jets)
        hf = m57_table_members(jets, HF_TABLE)
        after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
        prefix = graphed.weight(after)
        after = m57_weight(after, "hf", sf, hf)
        child = after[met.pt > MET_CUT]
        crossed = _at(prefix, child)
        lf = _members_at(m57_table_members(jets, LF_TABLE), child)

        def register_mu(target: Any, crossed: Any = crossed) -> Any:
            return m57_delta(target, "mu", crossed)

        def register_lf(target: Any, child: Any = child, sf: Any = sf, lf: Any = lf) -> Any:
            return m57_weight(target, "lf", _at(sf, child), lf)

        target: Any = child
        for step in (register_mu, register_lf) if delta_first else (register_lf, register_mu):
            target = step(target)
        expected = [
            m57_reindexed(m57_factor("pu", pu, m57_scaled(pu, PU)), child),
            m57_overlay("mu", m57_scaled(crossed, MU)),
            m57_joined(m57_reindexed(m57_factor("hf", sf, hf), child), "lf", lf),
        ]

        _agrees(session, graphed.weight(target), expected), delta_first
