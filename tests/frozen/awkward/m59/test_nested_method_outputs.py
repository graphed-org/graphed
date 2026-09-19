"""m59 integ-m59-M — a behavior method may return a NESTED tuple of arrays.

`metric_table(other, return_combinations=True)` answers `(metric, (a, b))`; today a tuple whose
members are not all arrays is refused wholesale. The nesting must come back as the same nesting of
`graphed.Array`s, while a single array, a flat tuple and a non-array leaf keep behaving as they do.
"""

from __future__ import annotations

import awkward as ak
import pytest
from m59_idiom_fixtures import eager_pairs, pair_session, recorded, tracer_pairs

from graphed import Array
from graphed.errors import GraphedTypeError


def test_a_nested_tuple_method_returns_the_same_nesting_of_arrays() -> None:
    session, pairs = pair_session()
    eager, abstract = eager_pairs().nested(), tracer_pairs().nested()

    metric, (left, right) = pairs.nested()

    for out, expected, concrete in (
        (metric, abstract[0], eager[0]),
        (left, abstract[1][0], eager[1][0]),
        (right, abstract[1][1], eager[1][1]),
    ):
        assert isinstance(out, Array)
        assert session.form(out).describe() == str(ak.Array(expected).type)
        assert ak.to_list(session.materialize(out)) == ak.to_list(concrete)


def test_single_and_flat_returns_record_exactly_as_before() -> None:
    session, pairs = pair_session()
    eager = eager_pairs()

    one, flat = pairs.one(), pairs.flat()

    assert isinstance(one, Array) and len(flat) == 2
    node = recorded(session, one)
    assert (node["kind"], node["name"], node["params"]) == (
        "op",
        "method",
        {"args": "[]", "kwargs": "{}", "method": "one"},
    )
    for index, leaf in enumerate(flat):
        leaf_node = recorded(session, leaf)
        assert (leaf_node["kind"], leaf_node["name"]) == ("op", "method")
        assert leaf_node["params"] == {"args": "[]", "index": index, "kwargs": "{}", "method": "flat"}
    assert ak.to_list(session.materialize(one)) == ak.to_list(eager.one())
    assert [ak.to_list(session.materialize(x)) for x in flat] == [ak.to_list(v) for v in eager.flat()]

    other, elsewhere = pair_session()
    assert session.serialized_ir(one) == other.serialized_ir(elsewhere.one())


def test_a_non_array_leaf_is_refused_while_the_same_nesting_of_arrays_records() -> None:
    _session, pairs = pair_session()

    with pytest.raises(GraphedTypeError) as info:
        pairs.leafy()

    assert "leafy():" in str(info.value) and "cannot be recorded" in str(info.value)
    assert isinstance(pairs.nested()[1][1], Array)  # same nesting, every leaf an array: records


def test_the_same_nested_call_interns_and_distinct_leaves_are_distinct_nodes() -> None:
    _session, pairs = pair_session()

    first, second = pairs.nested(), pairs.nested()

    ids = [first[0].node_id, first[1][0].node_id, first[1][1].node_id]
    assert ids == [second[0].node_id, second[1][0].node_id, second[1][1].node_id]
    assert len(set(ids)) == 3
