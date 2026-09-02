"""§5.4 spelled through `gak.join`, the awkward-idiom representative of the boundary refusal.

The m39/m40 plan builders are single-boundary: they take one output and pick one join/exchange node
out of the store, so a variation crossing a boundary compiles to a silent miscompilation. v1 refuses
the `Varied` OPERAND instead. What m49 freezes is the MESSAGE shape, worded over what the site knows
— the refusing VERB and the offending container's labels; there is no boundary node at an operand
check, and the container carries N labels, not one.

The positive control is what keeps this from freezing a blanket "a Varied near a Join raises": a
variation entirely DOWNSTREAM of the join compiles and produces correct results per universe, routed
through `Session.materialize` per universe (the m40 join fixtures' house route, not a plan builder).
"""

from __future__ import annotations

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.errors import GraphedError

ON = ["run", "lumi", "event"]


def _sources(session: Session) -> tuple[object, object]:
    """m40's shape: left row 0 matches nothing, rows 1-2 each match both right rows."""
    left = from_awkward(
        session,
        "left",
        ak.Array({"run": [1, 1, 1], "lumi": [1, 1, 1], "event": [1, 2, 2], "lv": [10.0, 20.0, 21.0]}),
    )
    right = from_awkward(
        session, "right", ak.Array({"run": [1, 1], "lumi": [1, 1], "event": [2, 2], "rv": [200.0, 201.0]})
    )
    return left, right


@pytest.mark.parametrize("grouped", [False, True])
def test_a_varied_operand_is_refused_naming_the_verb_and_the_containers_labels(grouped: bool) -> None:
    session = Session(AwkwardBackend())
    left, right = _sources(session)
    varied = graphed.vary(left, "jes", up=left, down=left)

    with pytest.raises(GraphedError) as excinfo:
        gak.join(varied, right, on=ON, grouped=grouped)

    message = str(excinfo.value)
    assert "gak.join" in message
    for label in graphed.labels(varied):
        assert label in message


def test_a_variation_downstream_of_the_join_compiles_and_answers_per_universe() -> None:
    session = Session(AwkwardBackend())
    left, right = _sources(session)
    joined = gak.join(left, right, on=ON, how="inner")
    scaled = graphed.vary(joined.lv, "sf", up=joined.lv * 1.1, down=joined.lv * 0.9)

    relational = sorted(ak.to_list(session.materialize(joined.lv)))
    assert relational == [20.0, 20.0, 21.0, 21.0]
    for label, scale in (("nominal", 1.0), ("sf_up", 1.1), ("sf_down", 0.9)):
        universe = sorted(ak.to_list(session.materialize(graphed.universe(scaled, label))))
        assert universe == pytest.approx([value * scale for value in relational])
