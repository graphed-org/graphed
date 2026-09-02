"""§2.1's one-row-space rule for `vary`'s INHERITED members (`vary._align`).

The container's handle is the mask-derived child, while the inherited members were read through the
parent, so each has to be re-indexed across the mask link — label-aligned per §2.4, which for a
`Varied` mask makes the aligned member a container over the MASK's labels.
"""

from __future__ import annotations

import numpy as np
from m49_vary_fixtures import (
    BTAG_SCALE,
    JES_SCALE,
    LOOSE_CUT,
    NOMINAL_CUT,
    VECTOR,
    varied_mask_context_program,
)

import graphed
from graphed.varied import Varied, member_of

MASK_REFERENCE = {"nominal": VECTOR > NOMINAL_CUT, "cut_lo": VECTOR > LOOSE_CUT}
SCALE = {"nominal": 1.0, "jes_up": JES_SCALE}


def test_inherited_members_are_re_indexed_into_the_containers_row_space() -> None:
    program = varied_mask_context_program()
    container = program.varied

    assert graphed.context_of(container) is program.selected
    for label, scale in SCALE.items():
        inherited = graphed.universe(container, label)
        assert isinstance(inherited, Varied)
        assert graphed.labels(inherited) == program.mask_labels
        for mask_label, mask in MASK_REFERENCE.items():
            got = program.session.materialize(member_of(inherited, mask_label))
            assert np.allclose(np.asarray(got), VECTOR[mask] * scale)


def test_the_new_member_keeps_the_row_space_it_was_read_in() -> None:
    """The member supplied AT the child context is already in its row space; §2.1 reduces a
    `Varied` member as supplied, so it stays flat over the mask's central universe."""
    program = varied_mask_context_program()
    added = graphed.universe(program.varied, "btag_up")

    assert not isinstance(added, Varied)
    got = program.session.materialize(added)
    assert np.allclose(np.asarray(got), VECTOR[MASK_REFERENCE["nominal"]] * BTAG_SCALE)
