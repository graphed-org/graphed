"""C3 / design §4.6, §8-d: at combination a container contributes the member whose point equals
`restrict(point(L), axes(container))`, else its `"nominal"` member.

Node ids are fixture-dependent — the same program numbers differently under `from_record` and
`from_awkward` — so every expectation here is a RELATION between two named candidate nodes, never a
pinned integer.
"""

from __future__ import annotations

import awkward as ak
from m52_projection_fixtures import JOINT_LABEL, ONE_AT_A_TIME_LABEL, joint_weight_program

import graphed


def test_a_joint_label_resolves_to_the_shifted_member_not_the_nominal_one() -> None:
    """§8-d. Today the joint label gets the `jes`-only container's NOMINAL member — wrong by the
    full size of the shift, the silent wrong number this arc exists to eliminate."""
    program = joint_weight_program()
    jes_only = program.jes_only

    assert graphed.points(jes_only) == {
        "nominal": {},
        "jes_up": {"jes": "up"},
        "jes_down": {"jes": "down"},
    }
    assert graphed.points(program.ambient)[JOINT_LABEL] == {"btag": "hf_up", "jes": "up"}

    shifted = graphed.universe(jes_only, "jes_up")
    central = graphed.nominal(jes_only)
    assert shifted.node_id != central.node_id  # the two candidate answers are distinct nodes

    chosen = graphed.member_of(jes_only, JOINT_LABEL)
    assert chosen.node_id == shifted.node_id
    assert chosen.node_id != central.node_id
    assert ak.to_list(program.session.materialize(chosen)) == ak.to_list(
        program.session.materialize(shifted)
    )

    # the one-at-a-time label restricts to the origin on this container, so it stays nominal
    diagonal = graphed.member_of(jes_only, ONE_AT_A_TIME_LABEL)
    assert diagonal.node_id == central.node_id


def test_the_fast_path_returns_the_containers_own_member_by_identity() -> None:
    """§4.6's theorem: a label the container carries is not a special case, since
    `keys(point(L)) ⊆ axes(C)` makes the restriction the identity and point→label is unique
    (§4.11-2). Object identity pins that the fast path was kept."""
    program = joint_weight_program()

    for container in (program.jes_only, program.ambient):
        for label in graphed.labels(container):
            assert graphed.member_of(container, label) is container._members[label]
