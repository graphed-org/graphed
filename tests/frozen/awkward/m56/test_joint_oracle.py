"""m56 §2 item 3 / §4 oracle: the joint's ambient weight is the product, in registration order, of
each factor's member at that point — the name-identity factor's own tag and the second family's
cross member, both on the jets that point shifts. The one-at-a-time universe takes the same product
at the point `{jes: u}`, where the second family contributes its central, not 1.
"""

from __future__ import annotations

import awkward as ak
from m56_fanout_fixtures import HF_SF, JES_SF, TAGS, m56_capstone, m56_sf

import graphed


def test_the_joint_weight_is_the_two_level_product_of_the_factors() -> None:
    program = m56_capstone()
    read = program.session.materialize

    for direction in TAGS:
        jets = program.jets[f"jes_{direction}"]
        for tag in (*TAGS, "nominal"):
            label = f"hf_{tag}__jes_{direction}" if tag != "nominal" else f"jes_{direction}"
            oracle = m56_sf(jets, JES_SF[direction]) * m56_sf(jets, HF_SF[tag])
            got = ak.to_list(read(graphed.member_of(program.weight, label)))
            assert got == ak.to_list(read(oracle)), label
