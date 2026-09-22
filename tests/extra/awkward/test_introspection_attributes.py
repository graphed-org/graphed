"""ak.Array's own descriptors (`fields`, `ndim`, ...) describe a deferred array; they are answered from
the record-time typetracer or refused, never recorded as a `field` op. A record field or a behavior
class that provides the same name keeps today's path."""

from __future__ import annotations

import functools
from typing import Any

import awkward as ak
import pytest

from graphed import BoundMethod, Session
from graphed.awkward import AwkwardBackend, from_awkward

EAGER = ("fields", "type", "typestr", "ndim", "is_tuple", "positional_axis")
REFUSED = sorted(
    name
    for name, value in vars(ak.Array).items()
    if not name.startswith("_")
    and isinstance(value, property | functools.cached_property)
    and name not in EAGER
)
RECORDS = ak.Array([[{"x": 1, "y": 2.0}], [], [{"x": 3, "y": 4.0}]])


class TaggedArray(ak.Array):  # type: ignore[misc]  # awkward ships no type stubs for subclassing
    @property
    def doubled(self) -> Any:
        return self.x * 2

    @property
    def nbytes(self) -> Any:  # a behavior providing an ak.Array name owns it
        return self.x * 8

    def scaled(self, k: int) -> Any:
        return self.x * k


def _session(data: ak.Array, behavior: dict[Any, Any] | None = None) -> tuple[Session, Any]:
    session = Session(AwkwardBackend(behavior=behavior))
    return session, from_awkward(session, "events", data)


def test_refused_population_is_live() -> None:
    assert {"layout", "mask", "nbytes", "attrs", "behavior", "named_axis"} <= set(REFUSED)


@pytest.mark.parametrize("data", [RECORDS, ak.Array([[(1, 2.0)], [], [(3, 4.0)]])], ids=["record", "tuple"])
@pytest.mark.parametrize("name", EAGER)
def test_eager_names_answer_like_the_data_and_record_nothing(data: ak.Array, name: str) -> None:
    session, arr = _session(data)
    before = session.node_count()
    got, want = getattr(arr, name), getattr(data, name)
    assert session.node_count() == before
    assert type(got) is type(want)
    if name == "type":  # the deferred length is unknown; everything below it is exact
        assert got.content == want.content
    elif name == "typestr":
        assert got.split(" * ", 1)[1] == want.split(" * ", 1)[1]
    else:
        assert got == want


def test_fields_membership_and_ndim_comparison() -> None:
    session, arr = _session(RECORDS)
    assert "x" in arr.fields
    assert (arr.ndim == 2) is True
    assert (arr.ndim == 1) is False
    assert session.node_count() == 1


@pytest.mark.parametrize("name", REFUSED)
def test_refused_names_raise_with_guidance(name: str) -> None:
    session, arr = _session(RECORDS)
    hint = "gak.mask" if name == "mask" else "materialize"
    with pytest.raises(AttributeError, match=rf"cannot answer '{name}'; .*{hint}"):
        getattr(arr, name)
    assert session.node_count() == 1


@pytest.mark.parametrize("name", ["type", "fields", "ndim"])
def test_a_record_field_named_like_an_ak_array_attribute_is_still_a_field(name: str) -> None:
    data = ak.Array([{name: 1, "other": 0}, {name: 3, "other": 0}])
    session, arr = _session(data)
    got = getattr(arr, name)
    assert session.node_count() == 2
    assert ak.to_list(session.materialize(got)) == [1, 3]


def test_behavior_members_classify_as_before() -> None:
    data = ak.with_name(RECORDS, "Tagged")
    session, arr = _session(data, {("*", "Tagged"): TaggedArray})
    assert isinstance(arr.scaled, BoundMethod)
    assert session.node_count() == 1
    expected = TaggedArray(data, behavior={("*", "Tagged"): TaggedArray})
    assert ak.to_list(session.materialize(arr.doubled)) == ak.to_list(expected.doubled)
    assert ak.to_list(session.materialize(arr.nbytes)) == ak.to_list(expected.nbytes)
    assert session.node_count() == 3
