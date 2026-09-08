"""m55 §2 item 2's stacking clause: adding a tag to the family already registered.

A container carrying only this family's labels is admitted whatever else the context varies, so a
new tag on the nominal stacks; the same container built on the varied collection carries the tags
already registered and is refused by `check_family`, before anything is minted — the refusal a
weaker rule would have spelled as a shape error, or missed entirely.
"""

from __future__ import annotations

from m55_lockstep_fixtures import (
    base_toy,
    equivalence,
    jes_context,
    refused,
    restacked_containers,
    stacking_program,
)

import graphed


def test_a_new_tag_on_the_nominal_stacks_onto_the_registered_family() -> None:
    varied, hand = equivalence(stacking_program, "Jet", "MET")

    labels = ("nominal", "jes_up", "jes_down", "jes_flat")
    assert set(varied) == {(collection, label) for collection in ("Jet", "MET") for label in labels}
    assert varied == hand
    assert len(set(varied.values())) == len(varied)


def test_a_container_re_offering_a_registered_tag_is_refused() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    containers = restacked_containers(ctx)
    assert set(graphed.labels(containers["Jet"])) == {"nominal", "jes_up", "jes_down", "jes_flat"}

    message = refused(toy, lambda: graphed.vary(ctx, "jes", collections=containers))
    for fragment in ("already registered under", "jes"):
        assert fragment in message, fragment
