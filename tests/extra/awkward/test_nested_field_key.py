"""A tuple key's field names select fields and its other members slice, as in awkward (#257)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gnano
from graphed.systematics import universe

S = slice(None)
EAGER = ak.Array(
    [
        {"a": {"b": [1.0, 2.0], "c": [3.0, 4.0]}, "d": {"b": [5.0, 6.0]}},
        {"a": {"b": [7.0], "c": [8.0]}, "d": {"b": [9.0]}},
    ]
)
EQUAL = [
    ("a", "b"),
    (S, "a"),
    ("a", S, 0),
    ("a", "b", S, 0),
    (S, "a", "b", 0),
    ("a", ["b", "c"]),
    (["a", "d"], S, 0),
]
AXIS0 = [("a", 0), (0, "a"), ("a", 1, "b")]


def _eager(value: Any) -> tuple[str, list[Any]]:
    value = ak.Array(value)
    return str(value.type), value.tolist()


def _src() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward("e", EAGER, session=session)


def test_the_repro_reads_a_nested_field() -> None:
    x = ak.zip({"a": ak.zip({"b": [1.0, 2.0]})}, depth_limit=1)
    src = from_awkward("e", x, session=graphed.new_session())
    assert graphed.materialize(src[("a", "b")]).tolist() == x[("a", "b")].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("key", EQUAL, ids=repr)
def test_a_tuple_key_with_fields_equals_eager(key: tuple[Any, ...]) -> None:
    session, src = _src()
    assert _eager(session.materialize(src[key])) == _eager(EAGER[key])


@pytest.mark.parametrize("key", AXIS0, ids=repr)
def test_a_member_consuming_axis_0_keeps_the_axis_0_refusal(key: tuple[Any, ...]) -> None:
    _session, src = _src()
    assert EAGER[key] is not None
    with pytest.raises(TypeError, match="must keep the partitioned axis 0 whole"):
        src[key]


def test_a_field_list_followed_by_a_field_is_refused_by_name() -> None:
    _session, src = _src()
    assert EAGER[["a", "d"], "b"].fields == ["a", "d"]
    with pytest.raises(TypeError, match="a field list followed by a field name"):
        src[["a", "d"], "b"]


@pytest.mark.parametrize("key", [(["a", "d"], ["b"]), ("a", ["b"], "c")], ids=repr)
def test_a_field_list_before_another_field_is_refused_with_the_whole_key(
    key: tuple[Any, ...],
) -> None:
    _session, src = _src()
    with pytest.raises(TypeError, match="followed by a field name or field list") as info:
        src[key]
    assert repr(key) in str(info.value)


@pytest.mark.parametrize("key", [("a", "b"), ("a", S, 0), ("a", ["b", "c"])], ids=repr)
def test_an_event_context_reads_a_nested_field_key_as_eager(key: tuple[Any, ...]) -> None:
    session = Session(AwkwardBackend())
    events = gnano.events(from_awkward("e", EAGER, session=session))
    assert _eager(session.materialize(events[key])) == _eager(EAGER[key])


def test_an_event_context_keeps_the_axis_0_refusal() -> None:
    session = Session(AwkwardBackend())
    events = gnano.events(from_awkward("e", EAGER, session=session))
    with pytest.raises(TypeError, match="must keep the partitioned axis 0 whole"):
        events[0, "a"]


def test_a_varied_reads_a_nested_field_key_per_universe() -> None:
    session, src = _src()
    up = ak.with_field(EAGER, ak.with_field(EAGER.a, EAGER.a.b * 2, "b"), "a")
    shifted = ak.with_field(src, ak.with_field(src.a, src.a.b * 2, "b"), "a")
    varied = graphed.vary(src, "s", points={"up": shifted})
    out = varied["a", "b", S, 0]
    for label, eager in (("nominal", EAGER), ("s_up", up)):
        assert _eager(session.materialize(universe(out, label))) == _eager(eager["a", "b", S, 0])
