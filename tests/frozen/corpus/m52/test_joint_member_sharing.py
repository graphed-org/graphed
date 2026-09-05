"""vary-m53 — design §2: the auto-minted joint members are the one-at-a-time expressions.

The fanout binds each joint label to the SAME `sf_hf_up` / `sf_hf_down` object as its one-at-a-time
sibling and lets the POINT pick the inner JES universe. Building an arithmetically equal but
distinct expression instead adds a node the M4 reducer may merge back, which the histogram builders
refuse at compile time with a message about optimizer merges rather than about points; the node-id
relations below pin the sharing that avoids it. No histogram is needed — this is the recorded
structure alone.
"""

from __future__ import annotations

from m52_corpus_fixtures import HF_JOINTS, joint_program

import graphed


def test_the_four_joint_members_are_the_same_expression_objects_as_the_two_one_at_a_time_ones() -> None:
    program = joint_program()
    weight = program.weight
    sources = {"hf_up": program.sf_hf_up, "hf_down": program.sf_hf_down}

    # the two candidate answers per joint label are distinct NODES, so the relations below can fail
    # in the direction they guard
    for source in sources.values():
        for inner in ("jes_up", "jes_down"):
            assert graphed.member_of(source, inner).node_id != graphed.nominal(source).node_id

    # a one-at-a-time b-tag label projects to the origin on the SF container's `jes` axis: nominal
    for flavour, source in sources.items():
        assert (
            graphed.member_of(weight, f"btag_{flavour}").node_id
            == graphed.nominal(source).node_id
        )

    # each auto-minted joint label reads that same container's inner JES universe — no second SF
    for label, (src_key, jes_universe) in HF_JOINTS.items():
        source = sources[src_key]
        member = graphed.member_of(weight, label)
        assert member.node_id == graphed.member_of(source, jes_universe).node_id
        assert member.node_id != graphed.nominal(source).node_id
