"""vary-m53 §2: a plain `vary` over dependent members mints the full grid, and every joint resolves
to the real cross node the graph holds — never the nominal one the pre-m53 collapse produced.

The three assertions separate the three ways the fanout could be wrong: the SET (a missing or extra
universe), the RESOLUTION (a joint pointing at nominal, the silent drop this arc exists to remove),
and the VALUES (two universes that share an array, which a bare label check would miss).
"""

from __future__ import annotations

import awkward as ak
from m53_fanout_fixtures import GRID_POINTS, JOINT_LABELS, JOINT_SOURCES, fanout_weight

import graphed


def test_the_default_is_the_full_grid_of_fifteen_universes() -> None:
    program = fanout_weight()

    assert graphed.points(program.weight) == GRID_POINTS
    assert set(graphed.labels(program.weight)) == set(GRID_POINTS)
    assert len(graphed.labels(program.weight)) == 15

    joints = [label for label in graphed.labels(program.weight) if "__" in label]
    assert sorted(joints) == sorted(JOINT_LABELS)  # exactly the eight machine-minted joints


def test_every_joint_resolves_to_the_real_cross_node_not_nominal() -> None:
    program = fanout_weight()
    weight = program.weight

    for joint, (src_key, jes_universe) in JOINT_SOURCES.items():
        source = program.members[src_key]
        cross = graphed.member_of(source, jes_universe)
        assert cross.node_id != graphed.nominal(source).node_id  # candidates are distinct nodes

        member = graphed.member_of(weight, joint)
        assert member.node_id == cross.node_id, joint
        assert member.node_id != graphed.nominal(source).node_id, joint


def test_the_fifteen_universes_materialize_distinctly() -> None:
    program = fanout_weight()
    weight, observable = program.weight, program.observable
    assert len(graphed.labels(weight)) == 15  # or the distinctness below is over the wrong set

    fingerprints = {}
    for label in graphed.labels(weight):
        obs = program.session.materialize(graphed.member_of(observable, label))
        wgt = program.session.materialize(graphed.member_of(weight, label))
        fingerprints[label] = (tuple(ak.to_list(obs)), tuple(ak.to_list(wgt)))

    assert len(set(fingerprints.values())) == len(fingerprints), (
        f"universes share an (observable, weight) array: {sorted(fingerprints)}"
    )
