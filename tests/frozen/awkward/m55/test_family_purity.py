"""m55 §2 item 2: a `Varied` collection member must carry exactly the family being registered —
`"nominal"` plus that family's labels — and is refused loudly otherwise.

One leg per way a container acquires a foreign label: built ON a collection another family already
varied (it inherits those labels), built with MEMBERS that read one (the loose form mints joint
labels for the cross terms), or varying a different family outright. The paired accept is the same
second family built on the context's nominal, which item 2 admits — without it a refusal that
rejected every container would pass all three.
"""

from __future__ import annotations

from m55_lockstep_fixtures import (
    base_toy,
    equivalence,
    inherited_container,
    jes_context,
    joint_container,
    other_family_container,
    refused,
    second_family_program,
)

import graphed


def test_a_container_inheriting_another_familys_labels_is_refused() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    container = inherited_container(ctx)
    assert set(graphed.labels(container)) > {"nominal", "jer_up", "jer_down"}  # jes came with it

    message = refused(toy, lambda: graphed.vary(ctx, "jer", collections={"Jet": container}))
    for fragment in ("Jet", "jer", "jes_up", "jes_down", "{tag: record}"):
        assert fragment in message, fragment


def test_a_container_whose_members_read_a_varied_collection_is_refused() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    container = joint_container(ctx)
    assert [label for label in graphed.labels(container) if "__" in label]  # the fanout ran

    message = refused(toy, lambda: graphed.vary(ctx, "jer", collections={"Jet": container}))
    for fragment in ("Jet", "jer", "jer_up__jes_up", "jer_down__jes_down", "{tag: record}"):
        assert fragment in message, fragment


def test_a_container_varying_a_different_family_is_refused() -> None:
    toy = base_toy()
    container = other_family_container(toy.ctx)

    message = refused(toy, lambda: graphed.vary(toy.ctx, "jes", collections={"Jet": container}))
    for fragment in ("Jet", "jes", "jer_up", "jer_down", "{tag: record}"):
        assert fragment in message, fragment


def test_the_same_second_family_built_on_the_nominal_is_accepted() -> None:
    varied, hand = equivalence(second_family_program, "Jet", "MET")

    labels = ("nominal", "jes_up", "jes_down", "jer_up", "jer_down")
    assert set(varied) == {(collection, label) for collection in ("Jet", "MET") for label in labels}
    assert varied == hand
    assert len(set(varied.values())) == len(varied)
