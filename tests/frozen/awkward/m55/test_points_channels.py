"""m55 §2 item 1's `points=` channels in the shift form.

A `Varied` member unpacks to dependency-free members, so there is no fan-out for a PLACEMENT to
prune and it would re-point the label instead: the placement channel is refused beside one, and the
message points at the hand form, the spelling that expresses such a program. The same call without
the placement registers the family, which is what makes the refusal about the placement rather than
about the container. The DECLARING channel was never accepted in the shift form and m55 does not
accept it — the regression control.
"""

from __future__ import annotations

from m55_lockstep_fixtures import (
    FLAT_FACTOR,
    base_toy,
    jes_containers,
    refused,
    rescale,
)

import graphed


def test_a_placement_beside_a_varied_member_is_refused_pointing_at_the_hand_form() -> None:
    toy = base_toy()
    containers = jes_containers(toy.ctx)

    message = refused(
        toy,
        lambda: graphed.vary(toy.ctx, "jes", collections=containers, points=[{"jes": "up"}]),
    )
    assert "{tag: record}" in message

    accepted = graphed.vary(toy.ctx, "jes", collections=containers)
    assert set(graphed.labels(accepted["Jet"])) == {"nominal", "jes_up", "jes_down"}


def test_the_declaring_points_channel_stays_refused_in_the_shift_form() -> None:
    """Regression control: `points=` as a mapping declares tags, which the shift form takes from the
    collection values instead. Passes on the pre-m55 tree, so a run in which it also failed would
    prove the harness dead rather than the feature absent."""
    toy = base_toy()
    containers = jes_containers(toy.ctx)

    message = refused(
        toy,
        lambda: graphed.vary(
            toy.ctx,
            "jes",
            collections=containers,
            points={"flat": rescale(toy.ctx.Jet, FLAT_FACTOR)},
        ),
    )
    for fragment in ("points=", "{tag: record}"):
        assert fragment in message, fragment
