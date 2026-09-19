"""m59 extra — a leading `...` must have an axis to absorb.

`[..., 0]` leaves the partitioned axis whole only while the receiver is deeper than the members
after the Ellipsis. On a flat array the Ellipsis absorbs nothing and the key indexes axis 0, which
is a boundary the MVP does not model; depth is a typing fact, so the backend's `subscript` rule
refuses it at record time.
"""

from __future__ import annotations

import awkward as ak
import pytest
from m59_idiom_fixtures import (
    DATA,
    PlainSource,
    concatenated,
    eager_form,
    merge,
    nothing,
    partitioned,
    recorded,
    rows,
    session_over,
    tracer,
)

import graphed
from graphed.core.execution import SequentialRunner
from graphed.errors import GraphedTypeError

DEPTH1 = ak.Array([1.0, 2.0, 3.0, 4.0])
DEPTH2 = ak.Array([[1.0, 2.0], [4.0], [8.0, 16.0, 32.0]])
DEPTH3 = ak.Array([[[1.0, 2.0], [4.0]], [[8.0]]])

#: keys whose members after the `...` address every axis the receiver has
CONSUMED: dict[str, tuple[ak.Array, tuple[object, ...]]] = {
    "depth 1, [..., 0]": (DEPTH1, (Ellipsis, 0)),
    "depth 1, [..., 1:]": (DEPTH1, (Ellipsis, slice(1, None))),
    "depth 2, [..., 0, 0]": (DEPTH2, (Ellipsis, 0, 0)),
    "depth 3, [..., 0, 0, 0]": (DEPTH3, (Ellipsis, 0, 0, 0)),
}

#: and the ones that still leave axis 0 whole — `None` adds an axis and addresses none
ADMITTED: dict[str, tuple[ak.Array, tuple[object, ...]]] = {
    "depth 2, [..., 0]": (DEPTH2, (Ellipsis, 0)),
    "depth 2, [..., None, 0]": (DEPTH2, (Ellipsis, None, 0)),
    "depth 3, [..., 0]": (DEPTH3, (Ellipsis, 0)),
    "depth 3, [..., 0, 0]": (DEPTH3, (Ellipsis, 0, 0)),
}


@pytest.mark.parametrize("label", list(CONSUMED))
def test_a_leading_ellipsis_that_absorbs_nothing_is_ill_typed(label: str) -> None:
    data, key = CONSUMED[label]
    session, root = session_over(data)
    before = session.node_count()

    with pytest.raises(GraphedTypeError, match=r"leading \.\.\."):
        root[key]

    assert session.node_count() == before


@pytest.mark.parametrize("label", list(ADMITTED))
def test_a_leading_ellipsis_with_an_axis_to_absorb_still_records_and_evaluates(label: str) -> None:
    data, key = ADMITTED[label]
    session, root = session_over(data)

    out = root[key]

    assert recorded(session, out)["kind"] == "op"
    assert session.form(out).describe() == eager_form(tracer(data)[key])
    assert ak.to_list(session.materialize(out)) == ak.to_list(data[key])


def test_a_partitioned_run_of_a_surviving_leading_ellipsis_equals_the_unpartitioned_one() -> None:
    value = {}
    for steps in (1, 2):
        source = PlainSource()
        session, root = partitioned(source)
        out = root.x[Ellipsis, 0]
        plan = graphed.aggregate_plan(out, reduce=rows, combine=merge, empty=nothing, steps_per_file=steps)
        assert recorded(session, out)["kind"] == "op"  # survived the guard, still fusible
        value[steps] = SequentialRunner().run(plan).value
        assert len(source.seen) == steps

    assert len(value[2]) == 2 and len(value[1]) == 1
    assert concatenated(value[2]) == concatenated(value[1]) == ak.to_list(DATA.x[Ellipsis, 0])
