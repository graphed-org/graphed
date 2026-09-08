"""m55 §2 item 5: every refusal lands before anything is minted, so a refused call is not a
poisoned family.

The other refusal tests each check the registry through `refused`; this one checks the CONSEQUENCE
the registry check exists for — a leaked binding would make the retry collide on the very labels the
refused call named, and the family would be unregisterable for the life of the Session.
"""

from __future__ import annotations

from m55_lockstep_fixtures import (
    JER_TAGS,
    base_toy,
    jer_containers,
    jes_context,
    joint_container,
    refused,
    spelled,
)

import graphed


def test_a_refused_call_leaves_the_family_registrable() -> None:
    toy = base_toy()
    ctx = jes_context(toy)
    container = joint_container(ctx)
    before = dict(toy.session._points)

    message = refused(toy, lambda: graphed.vary(ctx, "jer", collections={"Jet": container}))
    assert "jer_up__jes_up" in message  # the refusal fired on the labels, not on something upstream
    assert dict(toy.session._points) == before

    containers = jer_containers(ctx)
    registered = graphed.vary(ctx, "jer", collections=spelled("varied", containers, "jer", JER_TAGS))

    assert set(graphed.labels(registered["Jet"])) == {
        "nominal",
        "jes_up",
        "jes_down",
        "jer_up",
        "jer_down",
    }
    for collection in ("Jet", "MET"):
        for tag in JER_TAGS:
            label = f"jer_{tag}"
            assert (
                graphed.member_of(registered[collection], label).node_id
                == graphed.member_of(containers[collection], label).node_id
            ), (collection, label)
