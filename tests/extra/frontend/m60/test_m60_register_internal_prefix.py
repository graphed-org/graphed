"""m60 integ-m60-V — `register_internal` takes a prefix that can match a module.

A prefix that can match nothing would be a silent no-op in the one call that exists to fix silent
wrong provenance: `"lib."` is the same declaration as `"lib"`, and a prefix naming no module is
refused rather than registered.

Registration is process-global with no inverse, so this file owns the `m60x_lib` prefix: no frozen
leg registers it, and the sibling `m60x_libx` is what the dotted-component rule must leave alone.
"""

from __future__ import annotations

import os

import m60x_lib
import m60x_libx
import pytest
from m13_toy import ToyBackend, ToyForm

import graphed.provenance
from graphed import Array, Session


def toy_session() -> tuple[Session, Array]:
    session = Session(ToyBackend())
    return session, session.source("x", form=ToyForm("source"), data=None)


def test_a_trailing_dot_spelling_is_the_same_declaration() -> None:
    graphed.provenance.register_internal("m60x_lib.")
    session, x = toy_session()

    mine = m60x_lib.select(x)
    sibling = m60x_libx.select(mine)  # chained: one op over one input is ONE hash-consed node

    assert os.path.basename(session.provenance(mine).filename) == os.path.basename(__file__)
    assert session.provenance(mine).function == "test_a_trailing_dot_spelling_is_the_same_declaration"
    assert os.path.basename(session.provenance(sibling).filename) == "m60x_libx.py"


@pytest.mark.parametrize("spelling", ["", "."])
def test_a_prefix_naming_no_module_is_refused(spelling: str) -> None:
    before = graphed.provenance._SKIP

    with pytest.raises(ValueError):
        graphed.provenance.register_internal(spelling)

    assert before == graphed.provenance._SKIP  # nothing registered on the way out
