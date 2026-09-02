"""§5.4: a boundary/plan verb refuses a `Varied` operand, and a variation downstream of the
boundary still compiles per universe."""

from __future__ import annotations

import numpy as np
import pytest
from m49_vary_fixtures import JOINED_PT, JOINED_SF, join_program

import graphed
from graphed.errors import GraphedError

JES = {"jes_up": 1.1, "jes_down": 0.9}


def _varied_left(program: object) -> graphed.Varied:
    pt = program.left["pt"]
    return graphed.vary(pt, "jes", **{tag.split("_", 1)[1]: pt * factor for tag, factor in JES.items()})


@pytest.mark.parametrize("side", ["left", "right"])
def test_join_refuses_a_varied_operand_naming_the_verb_and_the_labels(side: str) -> None:
    program = join_program()
    varied = _varied_left(program)
    other = program.right if side == "left" else program.left

    with pytest.raises(GraphedError) as caught:
        if side == "left":
            graphed.join(varied, other, on=["key"])
        else:
            graphed.join(other, varied, on=["key"])

    message = str(caught.value)
    assert "graphed.join" in message
    for label in graphed.labels(varied):
        assert label in message


@pytest.mark.parametrize(
    ("verb", "call"),
    [
        ("graphed.repartition", lambda varied: graphed.repartition(varied, n=2)),
        ("graphed.pack_key", lambda varied: graphed.pack_key(varied, on=["key"])),
    ],
)
def test_every_boundary_verb_names_the_verb_and_the_labels(verb: str, call: object) -> None:
    program = join_program()
    varied = _varied_left(program)

    with pytest.raises(GraphedError) as caught:
        call(varied)

    message = str(caught.value)
    assert verb in message
    for label in graphed.labels(varied):
        assert label in message


def test_a_variation_downstream_of_the_join_still_compiles_per_universe() -> None:
    """The positive control: a blanket "a Varied near a Join raises" must fail the suite."""
    program = join_program()
    pt, sf = program.joined["pt"], program.joined["sf"]
    varied = graphed.vary(pt * sf, "jes", **{tag.split("_", 1)[1]: (pt * f) * sf for tag, f in JES.items()})

    expected = {"nominal": JOINED_PT * JOINED_SF} | {
        label: (JOINED_PT * factor) * JOINED_SF for label, factor in JES.items()
    }
    assert set(graphed.labels(varied)) == set(expected)
    for label, reference in expected.items():
        got = program.session.materialize(graphed.universe(varied, label))
        assert np.array_equal(np.asarray(got), reference)
