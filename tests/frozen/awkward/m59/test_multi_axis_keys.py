"""m59 integ-m59-I1/I2/I4 — a tuple key that leaves the partitioned axis whole.

Such a key is an ordinary per-row op: the recorded form is the one eager awkward infers for the
same key, execution equals eager bit-for-bit, the node is NOT a boundary, and a run over two
partitions equals the unpartitioned one. Every m59-new outcome is reached inside a test body, so
the suite collects against a tree with no m59 implementation and fails at RUN time.
"""

from __future__ import annotations

import awkward as ak
import pytest
from m59_idiom_fixtures import (
    ACCEPTED,
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


@pytest.mark.parametrize("over", ["field", "record"])
@pytest.mark.parametrize("label", list(ACCEPTED))
def test_a_multi_axis_key_records_and_executes_as_eager_awkward_does(label: str, over: str) -> None:
    key = ACCEPTED[label]
    session, events = session_over()
    root, data = (events.x, DATA.x) if over == "field" else (events, DATA)

    out = root[key]

    assert session.form(out).describe() == eager_form(tracer(data)[key])
    assert ak.to_list(session.materialize(out)) == ak.to_list(data[key])
    assert recorded(session, out)["kind"] == "op"  # I2: per-row, fusible — never a boundary


def test_a_partitioned_run_of_a_multi_axis_key_equals_the_unpartitioned_one() -> None:
    value = {}
    for steps in (1, 2):
        source = PlainSource()
        session, root = partitioned(source)
        out = root.x[:, :2]
        plan = graphed.aggregate_plan(
            out, reduce=rows, combine=merge, empty=nothing, steps_per_file=steps
        )
        assert recorded(session, out)["kind"] == "op"
        value[steps] = SequentialRunner().run(plan).value
        assert len(source.seen) == steps  # the source really was read once per partition

    assert len(value[2]) == 2 and len(value[1]) == 1
    assert {kind for kind, _part in value[2]} == {kind for kind, _part in value[1]}
    assert concatenated(value[2]) == concatenated(value[1]) == ak.to_list(DATA.x[:, :2])


def test_equal_keys_intern_and_two_builds_agree_byte_for_byte() -> None:
    session, events = session_over()

    assert events.x[:, :2].node_id == events.x[:, :2].node_id
    distinct = [
        events.x[:, :2].node_id,
        events.x[:, :3].node_id,
        events.x[:, ::2].node_id,
        events.x[:, 0].node_id,
        events.x[:, -1].node_id,
        events.x[:, :, None].node_id,
        events.x[:, None, :].node_id,
    ]
    assert len(set(distinct)) == len(distinct)

    other, elsewhere = session_over()
    assert session.serialized_ir(events.x[:, :2]) == other.serialized_ir(elsewhere.x[:, :2])


def test_every_key_accepted_today_records_the_same_op_params_and_boundary_flag() -> None:
    session, events = session_over()
    for out, expected in (
        (events["x"], ("op", "field", {"field": "x"})),
        (events[["x", "y"]], ("op", "fields", {"fields": "x,y"})),
        (events[1:3], ("reduction", "slice", {"start": 1, "stop": 3})),
        (events[::2], ("reduction", "slice", {"step": 2})),
        (events[2], ("reduction", "index", {"i": 2})),
        (events.x[events.x > 1.0], ("op", "getitem", {})),
    ):
        node = recorded(session, out)
        assert (node["kind"], node["name"], node["params"]) == expected

    assert ak.to_list(session.materialize(events[1:3])) == ak.to_list(DATA[1:3])
    assert ak.to_list(session.materialize(events.x[events.x > 1.0])) == ak.to_list(DATA.x[DATA.x > 1.0])
