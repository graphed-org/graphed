"""m55 §2 item 3: the container's nominal, reindexed to the target context, must BE that context's
central member of the collection — the node equality decides admission, not where the container was
built.

Two accepts fix that the rule is about nodes and not about lineage: a container built at the parent
still registers at a masked child and at a `vary`-link descendant, where the reindex lands on the
same node. Three refusals fix the other direction, each naming both nodes: a nominal moved off the
central member, a projected child whose central member is one universe of the parent's, and a
collection the context reads off the record, whose masked read is a different node from the masked
container. The last two refuse programs the hand unpack accepts, so each asserts the hand form still
does — a refusal placed in the shared path would fail there.
"""

from __future__ import annotations

from m55_lockstep_fixtures import (
    JER_TAGS,
    MUSC_TAGS,
    base_toy,
    cites_node,
    equivalence,
    event_mask,
    jer_containers,
    jes_context,
    masked_child_program,
    muon_containers,
    record_read_program,
    refused,
    rescaled_nominal_container,
    spelled,
    vary_descendant_program,
)

import graphed

JER_LABELS = ("nominal", "jes_up", "jes_down", "jer_up", "jer_down")


def test_a_parent_built_container_registers_at_a_masked_child() -> None:
    varied, hand = equivalence(masked_child_program, "Jet", "MET")

    assert set(varied) == {(collection, label) for collection in ("Jet", "MET") for label in JER_LABELS}
    assert varied == hand
    assert len(set(varied.values())) == len(varied)


def test_a_parent_built_container_registers_at_a_vary_link_descendant() -> None:
    varied, hand = equivalence(vary_descendant_program, "Jet", "MET")

    assert ("MET", "unclustered_up") in varied  # the descendant's own family is still carried
    assert varied == hand
    assert len(set(varied.values())) == len(varied)


def test_a_container_whose_nominal_was_rescaled_is_refused_naming_both_nodes() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    child = ctx[event_mask(ctx)]
    container = rescaled_nominal_container(ctx)

    supplied = graphed.reindex_to(graphed.nominal(container), child).node_id
    central = graphed.member_of(child["Jet"], "nominal").node_id
    assert supplied != central  # or the leg pins nothing

    message = refused(toy, lambda: graphed.vary(child, "jer", collections={"Jet": container}))
    assert "Jet" in message
    assert cites_node(message, supplied), message
    assert cites_node(message, central), message


def test_a_parent_built_container_is_refused_at_a_projected_child() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    projected = graphed.universe(ctx, "jes_up")
    containers = jer_containers(ctx)

    supplied = graphed.reindex_to(graphed.nominal(containers["Jet"]), projected).node_id
    central = graphed.member_of(projected["Jet"], "nominal").node_id
    assert supplied != central

    message = refused(
        toy, lambda: graphed.vary(projected, "jer", collections={"Jet": containers["Jet"]})
    )
    assert "Jet" in message
    assert cites_node(message, supplied), message
    assert cites_node(message, central), message

    hand = graphed.vary(projected, "jer", collections=spelled("hand", containers, "jer", JER_TAGS))
    assert set(graphed.labels(hand["Jet"])) == {"nominal", "jer_up", "jer_down"}


def test_a_record_read_collection_is_refused_at_a_masked_child_and_accepted_at_its_own() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    child = ctx[event_mask(ctx)]
    containers = muon_containers(ctx)

    supplied = graphed.reindex_to(graphed.nominal(containers["Muon"]), child).node_id
    central = graphed.member_of(child["Muon"], "nominal").node_id
    assert supplied != central  # the child re-reads the masked record, not the masked collection

    message = refused(toy, lambda: graphed.vary(child, "musc", collections=containers))
    assert "Muon" in message
    assert cites_node(message, supplied), message
    assert cites_node(message, central), message

    hand = graphed.vary(child, "musc", collections=spelled("hand", containers, "musc", MUSC_TAGS))
    assert set(graphed.labels(hand["Muon"])) == {"nominal", "musc_up", "musc_down"}

    varied, at_parent = equivalence(record_read_program, "Muon")
    assert varied == at_parent
    assert len(set(varied.values())) == len(varied)
