"""m55 §2 items 1 and 4: a `Varied` collection member is an unpack of the hand form, and the two
spellings compile to the same nodes.

The legs separate the ways this can be wrong: the container is REFUSED (item 1), it is accepted but
unpacked to the wrong members (item 4's node equality), the labels are right and the arrays wrong
(the materialized leg), or the widened shape rule admits something that is neither shape (the
control).
"""

from __future__ import annotations

from m55_lockstep_fixtures import (
    JES,
    JES_TAGS,
    base_toy,
    equivalence,
    headline_program,
    jes_containers,
    met_of,
    mixed_collections,
    mixed_program,
    refused,
    rescale,
    spelled,
    values,
)

import graphed

JES_LABELS = ("nominal", "jes_up", "jes_down")
BOTH = {(collection, label) for collection in ("Jet", "MET") for label in JES_LABELS}


def test_a_varied_jet_and_its_propagated_met_mint_the_hand_forms_nodes() -> None:
    containers = jes_containers(base_toy().ctx)
    # the two legs below are the two SPELLINGS, or the comparison compares a program with itself
    assert all(isinstance(v, graphed.Varied) for v in spelled("varied", containers, "jes", JES_TAGS).values())
    assert all(isinstance(v, dict) for v in spelled("hand", containers, "jes", JES_TAGS).values())

    varied, hand = equivalence(headline_program, "Jet", "MET")

    assert set(varied) == BOTH
    assert varied == hand
    assert len(set(varied.values())) == len(varied)  # six distinct universes, not a collapse


def test_the_jes_up_met_is_the_met_of_the_jes_up_jets() -> None:
    toy, registered = headline_program("varied")

    shifted = met_of(graphed.member_of(registered["Jet"], "jes_up"), toy.ctx.RawMET)
    up = values(toy, graphed.member_of(registered["MET"], "jes_up").pt)
    assert up == values(toy, shifted.pt)
    assert up != values(toy, graphed.nominal(registered["MET"]).pt)  # the shift really moved it


def test_a_tag_mapping_and_a_varied_mix_in_one_call() -> None:
    mixed = mixed_collections(jes_containers(base_toy().ctx), "varied")
    assert isinstance(mixed["Jet"], dict)  # one call, two shapes — or the leg is not a mix
    assert isinstance(mixed["MET"], graphed.Varied)

    varied, hand = equivalence(mixed_program, "Jet", "MET")

    assert set(varied) == BOTH
    assert varied == hand
    assert len(set(varied.values())) == len(varied)


def test_a_varied_whose_tags_differ_from_the_mapping_is_out_of_lockstep() -> None:
    toy = base_toy()
    containers = jes_containers(toy.ctx)

    message = refused(
        toy,
        lambda: graphed.vary(
            toy.ctx,
            "jes",
            collections={
                "Jet": {"up": rescale(toy.ctx.Jet, JES["up"])},
                "MET": containers["MET"],
            },
        ),
    )
    for fragment in ("lockstep", "Jet", "MET", "down"):
        assert fragment in message, fragment  # "down" only if the Varied was unpacked first


def test_a_value_that_is_neither_a_mapping_nor_a_varied_stays_refused() -> None:
    """Regression control: m55 widens the accepted collection values to two shapes, not to any
    value. Passes on the pre-m55 tree, so a run in which it also failed would prove the harness
    dead rather than the feature absent."""
    toy = base_toy()

    message = refused(
        toy, lambda: graphed.vary(toy.ctx, "jes", collections={"Jet": toy.ctx.Jet})
    )
    for fragment in ("Jet", "{tag: record}"):
        assert fragment in message, fragment
