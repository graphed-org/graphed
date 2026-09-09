"""m56 §2 item 2 / §4 order independence: registering the name-identity weight before or after the
family built over the shifted objects gives the same universes and, per universe, the same Jet, MET
and ambient weight. The ambient here has exactly two factors, so the product is bit-equal either way.
"""

from __future__ import annotations

import awkward as ak
from m56_fanout_fixtures import Capstone, m56_capstone, m56_joints, m56_minted

import graphed


def _universes(program: Capstone) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """{label: (Jet pt, MET pt, ambient weight)} materialized for every universe the program has."""
    jets, met = program.ctx["Jet"], program.ctx["MET"]
    read = program.session.materialize
    return {
        label: (
            ak.to_list(read(graphed.member_of(jets, label).pt)),
            ak.to_list(read(graphed.member_of(met, label).pt)),
            ak.to_list(read(graphed.member_of(program.weight, label))),
        )
        for label in graphed.labels(program.weight)
    }


def test_the_two_registration_orders_give_the_same_universes_and_values() -> None:
    weight_first = m56_capstone()
    family_first = m56_capstone(weight_first=False)

    # not an empty agreement: both orders carry the cross-kind joints
    assert m56_minted(weight_first.weight, "hf") == m56_joints("hf", "jes") | m56_joints("hf", "jer")
    assert _universes(weight_first) == _universes(family_first)
