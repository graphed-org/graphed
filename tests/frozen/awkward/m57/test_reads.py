"""m57 §2.1/§2.3: the handle's READ decides nothing on its own.

Every composition handed out is recorded the moment it exists — a fresh composition, a memo hit, the
member an adoption hands the child — so inserting or removing a `graphed.weight()` read between
registrations changes no decision and no value, and the order of two relative-delta registrations on
handles read at different times changes nothing either.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from m57_dedupe_fixtures import (
    HF_TABLE,
    LF_TABLE,
    MET_CUT,
    MU,
    NU,
    PU,
    TRIG,
    m57_ambient_values,
    m57_base,
    m57_by_label,
    m57_delta,
    m57_factor,
    m57_node,
    m57_oracle_values,
    m57_overlay,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_table_members,
    m57_trig,
    m57_weight,
)

import graphed


def test_a_handle_read_before_a_join_decides_and_values_as_one_read_after_it() -> None:
    """A join leaves the nominal member alone, so a handle read before it is still the composition
    it names: both read spellings give the same one-SF universes and the same nominal node."""

    def program(extra_read: bool) -> tuple[Any, Any, list[Any]]:
        session, ctx = m57_base()
        jets = ctx["Jet"]
        sf, pu = m57_sf(jets), m57_pu(jets)
        pu_members = m57_scaled(pu, PU)
        hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)
        after = m57_weight(m57_weight(ctx, "pu", pu, pu_members), "hf", sf, hf)
        handle = graphed.weight(after)
        after = m57_weight(after, "lf", sf, lf)  # the join: the nominal member is unchanged
        if extra_read:
            graphed.weight(after)
        after = m57_delta(after, "mu", handle)
        ops = [
            m57_factor("pu", pu, pu_members),
            m57_factor("hf", sf, hf, **m57_by_label("lf", lf)),
            m57_overlay("mu", m57_scaled(handle, MU)),
        ]
        return session, graphed.weight(after), ops

    plain_session, plain, plain_ops = program(extra_read=False)
    read_session, read, read_ops = program(extra_read=True)

    assert m57_ambient_values(plain_session, plain) == m57_oracle_values(
        plain_session, plain_ops, graphed.labels(plain)
    )
    assert m57_ambient_values(read_session, read) == m57_oracle_values(
        read_session, read_ops, graphed.labels(read)
    )
    assert m57_ambient_values(plain_session, plain) == m57_ambient_values(read_session, read)


def test_a_handle_read_over_a_prefix_anchors_after_that_prefix() -> None:
    """The overlay sits right after the last factor it was read over, so a factor registered after
    the read multiplies its result and keeps its own universe untouched."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    pu_members, trig_members = m57_scaled(pu, PU), m57_scaled(trig, TRIG)
    hf = m57_table_members(jets, HF_TABLE)

    after = m57_weight(m57_weight(ctx, "pu", pu, pu_members), "hf", sf, hf)
    handle = graphed.weight(after)
    after = m57_weight(after, "trig", trig, trig_members)
    after = m57_delta(after, "mu", handle)
    weight = graphed.weight(after)
    ops = [
        m57_factor("pu", pu, pu_members),
        m57_factor("hf", sf, hf),
        m57_overlay("mu", m57_scaled(handle, MU)),
        m57_factor("trig", trig, trig_members),
    ]

    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_two_relative_delta_families_agree_in_both_orders_and_prefix_lengths() -> None:
    """Each overlay is anchored by the composition its own handle names, so neither the registration
    order of the two nor how long the older handle's prefix is can move a universe."""

    def program(*, delta_first: bool, early: bool, extra_read: bool) -> tuple[Any, Any, list[Any]]:
        session, ctx = m57_base()
        jets = ctx["Jet"]
        sf, pu = m57_sf(jets), m57_pu(jets)
        pu_members = m57_scaled(pu, PU)
        hf = m57_table_members(jets, HF_TABLE)
        after = m57_weight(ctx, "pu", pu, pu_members)
        older = graphed.weight(after) if early else None
        after = m57_weight(after, "hf", sf, hf)
        if older is None:
            older = graphed.weight(after)
        current = graphed.weight(after)

        def register_mu(target: Any) -> Any:
            return m57_delta(target, "mu", older, MU)

        def register_nu(target: Any) -> Any:
            return m57_delta(target, "nu", current, NU)

        steps: tuple[Callable[[Any], Any], ...] = (
            (register_mu, register_nu) if delta_first else (register_nu, register_mu)
        )
        for step in steps:
            if extra_read:
                graphed.weight(after)
            after = step(after)
        mu_op = m57_overlay("mu", m57_scaled(older, MU))
        nu_op = m57_overlay("nu", m57_scaled(current, NU))
        ops = (
            [m57_factor("pu", pu, pu_members), mu_op, m57_factor("hf", sf, hf), nu_op]
            if early
            else [m57_factor("pu", pu, pu_members), m57_factor("hf", sf, hf), mu_op, nu_op]
        )
        return session, graphed.weight(after), ops

    for early in (True, False):
        runs = [
            program(delta_first=first, early=early, extra_read=read)
            for first in (True, False)
            for read in (True, False)
        ]
        expected = m57_oracle_values(runs[0][0], runs[0][2], graphed.labels(runs[0][1]))
        assert set(expected) > {"mu_up", "nu_up"}  # both families really are on the ambient
        for session, weight, _ops in runs:
            assert m57_ambient_values(session, weight) == expected, early


def test_an_overlay_insertion_leaves_a_whole_list_handle_fresh() -> None:
    """Inserting an overlay widens no factor's nominal member, so a handle read over the whole list
    before that registration is still fresh after it — the relative-delta family it nominates is
    accepted and reads the composition it named."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    pu_members = m57_scaled(pu, PU)
    hf = m57_table_members(jets, HF_TABLE)
    after = m57_weight(ctx, "pu", pu, pu_members)
    one_factor = graphed.weight(after)
    after = m57_weight(after, "hf", sf, hf)
    whole = graphed.weight(after)

    after = m57_delta(after, "mu", one_factor)  # an overlay inserted among the factors
    after = m57_delta(after, "nu", whole, NU)  # the older whole-list handle: still fresh
    weight = graphed.weight(after)
    ops = [
        m57_factor("pu", pu, pu_members),
        m57_overlay("mu", m57_scaled(one_factor, MU)),
        m57_factor("hf", sf, hf),
        m57_overlay("nu", m57_scaled(whole, NU)),
    ]

    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_at_a_masked_child_the_adopted_read_anchors_the_overlay_after_the_head() -> None:
    """The adoption records the adopted member as the child's own read, so a handle read at the child
    before anything is registered there anchors right after the head — with a factor registered at
    the child after it multiplying its result."""
    session, ctx = m57_base()
    jets, met = ctx["Jet"], ctx["MET"]
    sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
    pu_members, trig_members = m57_scaled(pu, PU), m57_scaled(trig, TRIG)
    hf = m57_table_members(jets, HF_TABLE)
    after = m57_weight(m57_weight(ctx, "pu", pu, pu_members), "hf", sf, hf)
    child = after[met.pt > MET_CUT]
    adopted = graphed.weight(child)  # a memo hit on the adopted container
    child_trig = graphed.reindex_to(trig, child)

    registered = m57_weight(child, "trig", child_trig, trig_members)
    registered = m57_delta(registered, "mu", adopted)
    weight = graphed.weight(registered)
    ops = [
        m57_factor("head", adopted, {}),
        m57_overlay("mu", m57_scaled(adopted, MU)),
        m57_factor("trig", child_trig, m57_scaled(child_trig, TRIG)),
    ]

    assert m57_ambient_values(session, weight) == m57_oracle_values(session, ops, graphed.labels(weight))


def test_at_the_nominal_projection_a_child_factor_makes_both_orders_agree() -> None:
    """At `graphed.nominal(ctx)` the read returns the adopted member itself; with a factor registered
    at the projection, the two registration orders of the crossed handle and that factor agree on
    the label set and on every value."""

    def program(delta_first: bool) -> tuple[Any, Any]:
        session, ctx = m57_base()
        jets, met = ctx["Jet"], ctx["MET"]
        sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
        hf = m57_table_members(jets, HF_TABLE)
        after = m57_weight(m57_weight(ctx, "pu", pu, m57_scaled(pu, PU)), "hf", sf, hf)
        handle = graphed.weight(after)
        projected = graphed.nominal(after)
        flat_trig = graphed.reindex_to(trig, projected)

        def register_mu(target: Any) -> Any:
            return m57_delta(target, "mu", handle)

        def register_trig(target: Any) -> Any:
            return m57_weight(target, "trig", flat_trig, m57_scaled(flat_trig, TRIG))

        steps: tuple[Callable[[Any], Any], ...] = (
            (register_mu, register_trig) if delta_first else (register_trig, register_mu)
        )
        target: Any = projected
        for step in steps:
            target = step(target)
        return session, graphed.weight(target)

    delta_session, delta_first = program(delta_first=True)
    factor_session, factor_first = program(delta_first=False)

    assert set(graphed.labels(delta_first)) == set(graphed.labels(factor_first))
    assert {"mu_up", "trig_up"} <= set(graphed.labels(delta_first))
    assert m57_ambient_values(delta_session, delta_first) == m57_ambient_values(factor_session, factor_first)


def test_without_a_child_factor_the_orders_label_sets_differ_but_every_shared_value_agrees() -> None:
    """At the nominal projection the crossed handle's members carry the parent's labels, which the
    projected context no longer composes, so the two orders' label sets differ while every label
    they share has the same value."""

    def program(delta_first: bool) -> tuple[Any, Any]:
        session, ctx = m57_base()
        jets = ctx["Jet"]
        sf, pu = m57_sf(jets), m57_pu(jets)
        hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)
        after = m57_weight(m57_weight(ctx, "pu", pu, m57_scaled(pu, PU)), "hf", sf, hf)
        handle = graphed.weight(after)
        projected = graphed.nominal(after)
        flat_sf = graphed.reindex_to(sf, projected)

        def register_mu(target: Any) -> Any:
            return m57_delta(target, "mu", handle)

        def register_lf(target: Any) -> Any:
            return m57_weight(
                target,
                "lf",
                flat_sf,
                {tag: graphed.reindex_to(member, projected) for tag, member in lf.items()},
            )

        steps: tuple[Callable[[Any], Any], ...] = (
            (register_mu, register_lf) if delta_first else (register_lf, register_mu)
        )
        target: Any = projected
        for step in steps:
            target = step(target)
        return session, graphed.weight(target)

    delta_session, delta_first = program(delta_first=True)
    other_session, other_first = program(delta_first=False)
    shared = set(graphed.labels(delta_first)) & set(graphed.labels(other_first))
    delta_values = m57_ambient_values(delta_session, delta_first)
    other_values = m57_ambient_values(other_session, other_first)

    assert {"mu_up", "lf_up"} <= shared
    assert {label: delta_values[label] for label in shared} == {
        label: other_values[label] for label in shared
    }


def test_an_ancestor_factor_join_after_the_overlay_leaves_its_universe_unchanged() -> None:
    """The expansion rebuilds the child's list from provenance, so an overlay anchored after the head
    stays before a factor the child registered earlier and its universe does not change."""
    for projected in (False, True):
        session, ctx = m57_base()
        jets, met = ctx["Jet"], ctx["MET"]
        sf, pu, trig = m57_sf(jets), m57_pu(jets), m57_trig(met)
        hf, lf = m57_table_members(jets, HF_TABLE), m57_table_members(jets, LF_TABLE)
        after = m57_weight(m57_weight(ctx, "pu", pu, m57_scaled(pu, PU)), "hf", sf, hf)
        child = graphed.nominal(after) if projected else after[met.pt > MET_CUT]
        adopted = graphed.weight(child)
        child_trig = graphed.reindex_to(trig, child)
        after_trig = m57_weight(child, "trig", child_trig, m57_scaled(child_trig, TRIG))
        with_overlay = m57_delta(after_trig, "mu", adopted)
        before = m57_ambient_values(session, graphed.weight(with_overlay))

        expanded = m57_weight(
            with_overlay,
            "lf",
            graphed.reindex_to(sf, child),
            {tag: graphed.reindex_to(member, child) for tag, member in lf.items()},
        )
        after_values = m57_ambient_values(session, graphed.weight(expanded))

        assert after_values["mu_up"] == before["mu_up"], projected
        assert after_values["mu_down"] == before["mu_down"], projected
        assert m57_node(graphed.weight(expanded)) == m57_node(graphed.weight(with_overlay)), projected
