"""m57 implementation-review closures: the outcomes the frozen suite states but cannot fail on.

Each leg is the reviewer's own program, and each kills a one-hunk mutant of `graphed.systematics.
ambient` the whole frozen tree survives: the widening check compared one level too deep (C-B), the
covered-READ refusal deleted (C-A), the identity chain keeping only the last row space (C-C), the
overlay-anchoring skip on the expanded head (C-H), `ambient_entries` sorted by slot (C-I), a
left-out overlay kept through an expansion (C-J), and the mask identity `_crossed` appends (C-K).
"""

from __future__ import annotations

from typing import Any

import pytest
from m57_dedupe_fixtures import (
    EVENTS,
    HF_TABLE,
    JES,
    LF_TABLE,
    MET_CUT,
    MF_TABLE,
    MU,
    NU,
    PU,
    TRIG,
    m57_base,
    m57_delta,
    m57_entries,
    m57_explain,
    m57_node_count,
    m57_pu,
    m57_scaled,
    m57_sf,
    m57_shifted,
    m57_table_members,
    m57_values,
    m57_weight,
)

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.context import EventContext
from graphed.errors import GraphedError


def _shape(ctx: Any) -> list[tuple[str, tuple[str, ...]]]:
    """The ambient's operations as the user's instrument reports them: kind and families, in the
    order the composition applies them."""
    return [(rider.kind, tuple(rider.families)) for _slot, rider, _entry in m57_entries(ctx)]


def _slots(ctx: Any) -> list[int]:
    """The slot of each operation in composition order (`_SLOT` is process-wide, so only the
    ORDER of these is an invariant, never their values)."""
    return [slot for slot, _rider, _entry in m57_entries(ctx)]


# ---- C-B: the widening check compares the central and the entry at one level -----------------
def _widening_program() -> tuple[Session, Any, Any, Any]:
    """§2.1's blessed spelling: a `jes` shift, `hf` over the SF computed on the SHIFTED jets, `pu`,
    the overlay `mu` over both, then a nominal projection and a join of the ancestor's `pu` there —
    which re-indexes the head. Returns what a further family needs to name `hf`'s central."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    varied_sf = m57_sf(sjets)  # the central CARRIES the jes universes
    pu = m57_pu(graphed.nominal(sjets))
    after = m57_weight(shifted, "hf", varied_sf, m57_table_members(sjets, HF_TABLE))
    after = m57_weight(after, "pu", pu, m57_scaled(pu, PU))
    after = m57_delta(after, "mu", graphed.weight(after))
    flat = graphed.nominal(after)
    joined = m57_weight(flat, "pu", pu, m57_scaled(pu, PU))
    return session, joined, sjets, varied_sf


def test_a_central_built_above_a_nominal_projection_joins_as_its_re_indexed_spelling_does() -> None:
    """The re-index the registration performs peels the central to the projected level, so the
    union adds nothing and the join stands — §2.1's "a central built at an ancestor still names its
    factor after the expansion". Comparing the RAW central against the entry's member refuses the
    parent's spelling while accepting `reindex_to`, deciding one registration two ways."""
    outcomes = []
    for spelling in ("as built", "reindexed"):
        session, joined, sjets, varied_sf = _widening_program()
        central = varied_sf if spelling == "as built" else graphed.reindex_to(varied_sf, joined)
        minted = m57_node_count(session)
        registered = m57_weight(joined, "lf", central, m57_table_members(sjets, LF_TABLE))
        weight = graphed.weight(registered)
        outcomes.append(
            (
                graphed.labels(weight),
                m57_values(session, graphed.member_of(weight, "nominal")),
                _shape(registered),
                m57_node_count(session) - minted,
            )
        )

    assert outcomes[0] == outcomes[1]
    labels, nominal, shape, _minted = outcomes[0]
    # the union added NOTHING: no jes label appears, which is why the refusal's premise was false
    assert labels == ("nominal", "pu_up", "pu_down", "lf_up", "lf_down")
    assert nominal == [3.0, 22.0, 4.0, 3.0, 3.5, 36.0]
    assert shape == [("factor", ("hf", "lf", "pu")), ("overlay", ("mu",))]


def test_a_join_that_genuinely_widens_a_covered_factor_is_still_refused() -> None:
    """The same check at the PARENT, where no projection has peeled anything: there the union DOES
    add the shift's universes under the overlay, and the refusal must still fire."""
    session, ctx = m57_base()
    shifted = m57_shifted(ctx, "jes", JES)
    sjets = shifted["Jet"]
    flat_jets = graphed.nominal(sjets)
    flat, pu = m57_sf(flat_jets), m57_pu(flat_jets)
    after = m57_weight(shifted, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", flat, m57_table_members(flat_jets, HF_TABLE))
    with_overlay = m57_delta(after, "mu", graphed.weight(after))
    widening = m57_sf(sjets)
    members = m57_table_members(sjets, LF_TABLE)
    minted = m57_node_count(session)  # after the members are built: the CALL is what is measured

    with pytest.raises(GraphedError) as caught:
        m57_weight(with_overlay, "lf", widening, members)

    assert "jes_up" in str(caught.value) and "'mu'" in str(caught.value)
    assert m57_node_count(session) == minted


# ---- C-A: the covered-READ refusal --------------------------------------------------------
def test_a_handle_over_a_prefix_of_a_covered_family_is_refused_inside_its_own_universe() -> None:
    """§2.3: a `graphed.weight()` handle read over a STRICT PREFIX of the factors a relative-delta
    family covers, named inside that family's own universe, is refused — nothing there re-derives
    the composition its members already are. The refusal names both the family and the universe,
    and leaves not one node behind."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    prefix_handle = graphed.weight(after)  # a read over {pu} ALONE
    after = m57_weight(after, "hf", sf, m57_scaled(sf, PU))
    whole = graphed.weight(after)  # the read over {pu, hf} the overlay covers
    after = m57_delta(after, "mu", whole)
    inside = graphed.universe(after, "mu_up")  # the overlay's OWN universe
    members = m57_scaled(prefix_handle, PU)
    minted = m57_node_count(session)  # after the members are built: the CALL is what is measured

    with pytest.raises(GraphedError) as caught:
        m57_weight(inside, "lf", prefix_handle, members)

    message = str(caught.value)
    assert "'mu'" in message and "'mu_up'" in message
    assert "strict prefix" in message
    assert m57_node_count(session) == minted


# ---- C-C: the identity chain keeps the nominal it had in EVERY row space ---------------------
def test_a_central_as_built_at_the_ancestor_names_its_factor_after_an_expansion() -> None:
    """A central handed to `vary()` AS BUILT at the ancestor — no `reindex_to`, which the chain's
    every frozen leg spells — still names its factor once the mask's expansion has re-indexed the
    entry. Keeping only the LAST row space's nominal makes it a second factor and SQUARES the SF."""
    session, ctx = m57_base()
    jets = ctx["Jet"]
    sf = m57_sf(jets)
    parent = m57_weight(ctx, "hf", sf, m57_table_members(jets, HF_TABLE))
    child = parent[parent["MET"].pt > MET_CUT]
    here = graphed.reindex_to(sf, child)
    lf = {tag: graphed.reindex_to(m, child) for tag, m in m57_table_members(jets, LF_TABLE).items()}
    after = m57_weight(child, "lf", here, lf)  # the expansion
    mf = {tag: graphed.reindex_to(m, child) for tag, m in m57_table_members(jets, MF_TABLE).items()}

    joined = m57_weight(after, "mf", sf, mf)  # the central AS BUILT at the ancestor

    assert _shape(joined) == [("factor", ("hf", "lf", "mf"))]
    nominal = m57_values(session, graphed.member_of(graphed.weight(joined), "nominal"))
    assert nominal == m57_values(session, here)
    assert nominal != m57_values(session, here * here)


# ---- C-H: an overlay anchors AFTER an overlay already anchored there -------------------------
def test_a_second_overlay_on_the_same_prefix_read_anchors_after_the_first() -> None:
    """§2.1's "after any overlay already anchored there", on the path where the read ends inside an
    adopted head the registration must expand: the second delta lands behind the first, not between
    it and the factors it was read over."""
    _session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    handle = graphed.weight(after)  # a read over [pu] ALONE
    assert handle is not None
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    after = m57_delta(after, "mu", handle, MU)
    child = after[after["MET"].pt > MET_CUT]  # a row-space link: the head is adopted

    child = m57_delta(child, "nu", graphed.reindex_to(handle, child), NU)

    assert _shape(child) == [
        ("factor", ("pu",)),
        ("overlay", ("mu",)),
        ("overlay", ("nu",)),
        ("factor", ("hf",)),
    ]


# ---- C-I: `ambient_entries` lists COMPOSITION order, not slot order -------------------------
def test_ambient_entries_keep_composition_order_where_it_differs_from_slot_order() -> None:
    """An overlay anchors after the operations its handle was read over — BEFORE a factor
    registered later, whose slot is minted earlier. The listing and `explain` report the order the
    composition applies, so here they are NOT the registration order."""
    _session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", sf, m57_scaled(sf, PU))
    handle = graphed.weight(after)
    trig = m57_pu(jets) * 0.5
    after = m57_weight(after, "trig", trig, m57_scaled(trig, TRIG))
    after = m57_delta(after, "mu", handle)

    slots = _slots(after)
    assert [families for _kind, families in _shape(after)] == [("pu",), ("hf",), ("mu",), ("trig",)]
    assert slots != sorted(slots)  # the two orders genuinely differ here
    listed = [
        line.split(": ")[1].split("[")[0]
        for line in str(m57_explain(after)).splitlines()
        if line.startswith("  #")
    ]
    assert listed == ["pu", "hf", "mu", "trig"]


# ---- C-J: a left-out overlay is not resurrected by an expansion -----------------------------
def test_an_overlay_a_projection_left_out_is_not_brought_back_by_an_expansion() -> None:
    """A projection into a universe carrying none of an overlay's coordinates leaves the overlay
    out; a later join that EXPANDS the adopted head re-indexes the operations it stands for, and the
    left-out one must not reappear among them."""
    _session, ctx = m57_base()
    jets = ctx["Jet"]
    sf, pu = m57_sf(jets), m57_pu(jets)
    after = m57_weight(ctx, "pu", pu, m57_scaled(pu, PU))
    after = m57_weight(after, "hf", sf, m57_table_members(jets, HF_TABLE))
    handle = graphed.weight(after)
    after = m57_delta(after, "mu", handle)
    inside = graphed.universe(after, "hf_up")  # carries no mu coordinate
    assert _shape(inside) == [("factor", ("pu", "hf"))]

    joined = m57_weight(inside, "lf", pu, m57_scaled(pu, PU))  # joins pu: expands the head

    assert _shape(joined) == [("factor", ("pu", "lf", "hf"))]


# ---- C-K: the mask identity `_crossed` appends ----------------------------------------------
def test_a_central_spelled_as_the_re_indexed_node_names_its_factor_at_the_masked_child() -> None:
    """The entry's nominal in the NEW row space is an identity of its own, which a central built
    there names. A context-free central is the spelling that needs it: nothing records it as an
    ancestor's node re-indexed, so peeling the mask off the op record cannot reach it, and only the
    identity the expansion appended answers. Without that append the masked node is a SECOND factor
    over an already-masked value."""
    session = Session(AwkwardBackend())
    root = from_awkward(session, "events", EVENTS)
    ctx = EventContext(session, root, collections={"Jet": root.Jet, "MET": root.MET})
    free = 1.0 + 0.25 * gak.num(root.Jet)  # off the ROOT array: it carries no context
    after = graphed.vary(ctx, "hf", free, is_weight=True, points=m57_scaled(free, PU))
    child = after[after["MET"].pt > MET_CUT]
    joined = graphed.vary(child, "lf", free, is_weight=True, points=m57_scaled(free, TRIG))
    assert _shape(joined) == [("factor", ("hf", "lf"))]
    here = free[graphed.selection(child)]  # the very node the expansion re-indexed the entry to

    registered = graphed.vary(joined, "mf", here, is_weight=True, points=m57_scaled(here, MF_TABLE))

    assert _shape(registered) == [("factor", ("hf", "lf", "mf"))]
    weight = graphed.weight(registered)
    assert m57_values(session, graphed.member_of(weight, "nominal")) == m57_values(session, here)
