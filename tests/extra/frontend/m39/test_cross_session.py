"""Sources as named slots (one form and one data object per name), and ``graphed.materialize``
over several Sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, cast

import pytest

import graphed
from graphed import Session
from graphed.backend import Form
from graphed.core import Partition
from graphed.core.execution import SequentialRunner


@dataclass(frozen=True)
class _ToyForm:
    tag: str

    def describe(self) -> str:
        return self.tag


class SumBackend:
    """``add`` sums its inputs elementwise and counts every evaluation."""

    identity = "sum/0"
    calls: ClassVar[list[str]] = []

    def op_form(self, op: str, inputs: Sequence[Form], params: Mapping[str, object]) -> Form:
        return _ToyForm(op)

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        SumBackend.calls.append(op)
        if op == "add":
            return [sum(vs) for vs in zip(*cast("list[list[int]]", inputs), strict=True)]
        return inputs[0]

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> None:
        return None


@dataclass(frozen=True)
class Loader:
    """A lazy source; every load is counted."""

    values: tuple[int, ...]
    loads: ClassVar[list[tuple[int, ...]]] = []

    def __call__(self) -> list[int]:
        Loader.loads.append(self.values)
        return list(self.values)


@dataclass(frozen=True)
class Rows:
    """A partitioned source over in-memory rows."""

    data: tuple[int, ...]

    def __call__(self) -> list[int]:
        raise AssertionError("a plan never loads the whole dataset")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        uri = "toy://" + ",".join(map(str, self.data))
        return tuple(Partition.blind(uri, "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> list[int]:
        part = partition.resolve(len(self.data))
        return list(self.data[part.entry_start : part.entry_stop])


def _form() -> Form:
    return _ToyForm("f")


@pytest.fixture(autouse=True)
def _reset() -> None:
    SumBackend.calls.clear()
    Loader.loads.clear()


# ---- a name is a slot of one form bound to one data object -----------------------------------
def test_a_name_keeps_one_form_whatever_the_params() -> None:
    s = Session(SumBackend())
    data = Loader((1,))
    s.source("x", form=_form(), data=data)
    for params in ({}, {"uri": "other"}):
        with pytest.raises(ValueError, match="already a slot of another form"):
            s.source("x", form=_ToyForm("g"), data=data, **params)


def test_a_name_keeps_its_data_object() -> None:
    s = Session(SumBackend())
    data = Loader((1, 2))
    first = s.source("x", form=_form(), data=data)
    assert s.source("x", form=_form(), data=data).node_id == first.node_id
    for other in (Loader((3, 4)), Loader((1, 2))):  # another object, equal or not
        with pytest.raises(ValueError, match="already bound to other data"):
            s.source("x", form=_form(), data=other)


# ---- graphed.materialize ------------------------------------------------------------------------
def test_materialize_evaluates_arrays_of_several_sessions_in_their_shape() -> None:
    s1, s2 = Session(SumBackend()), Session(SumBackend())
    x = s1.source("x", form=_form(), data=Loader((1, 2)))
    y = s2.source("y", form=_form(), data=Loader((10, 20)))
    doubled = s2.record_op("add", [y, y])
    assert graphed.materialize(x) == [1, 2]
    assert graphed.materialize([x, doubled, "kept"]) == [[1, 2], [20, 40], "kept"]
    assert graphed.materialize((doubled,)) == ([20, 40],)
    assert graphed.materialize({"x": x, "y": doubled}) == {"x": [1, 2], "y": [20, 40]}


class _Pair(NamedTuple):
    first: Any
    second: Any


def test_materialize_evaluates_arrays_nested_in_collections() -> None:
    s1, s2 = Session(SumBackend()), Session(SumBackend())
    x = s1.source("x", form=_form(), data=Loader((1, 2)))
    y = s2.source("y", form=_form(), data=Loader((10, 20)))
    nested = {"a": [x, (y, {"b": _Pair(x, "kept")})]}
    result = graphed.materialize(nested)
    assert result == {"a": [[1, 2], ([10, 20], {"b": _Pair([1, 2], "kept")})]}
    assert type(result["a"][1][1]["b"]) is _Pair
    assert type(result["a"][1]) is tuple


_X, _Y = [1, 2], [10, 20]


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda x, y: x, _X),
        (lambda x, y: None, None),
        (lambda x, y: [], []),
        (lambda x, y: (), ()),
        (lambda x, y: {}, {}),
        (lambda x, y: [x], [_X]),
        (lambda x, y: (y,), (_Y,)),
        (lambda x, y: [[[[x]]]], [[[[_X]]]]),
        (lambda x, y: [(x, y), [y, x]], [(_X, _Y), [_Y, _X]]),
        (lambda x, y: {1: x, (2, 3): y}, {1: _X, (2, 3): _Y}),
        (lambda x, y: {"x": x, "y": [y], "z": {"w": (x, 3)}}, {"x": _X, "y": [_Y], "z": {"w": (_X, 3)}}),
    ],
    ids=[
        "array",
        "none",
        "empty-list",
        "empty-tuple",
        "empty-dict",
        "list-of-one",
        "tuple-of-one",
        "deep-list",
        "tuple-and-list",
        "non-str-keys",
        "mixed-depths",
    ],
)
def test_materialize_returns_the_structure_it_was_given(build: Any, expected: Any) -> None:
    s1, s2 = Session(SumBackend()), Session(SumBackend())
    x = s1.source("x", form=_form(), data=Loader(tuple(_X)))
    y = s2.source("y", form=_form(), data=Loader(tuple(_Y)))
    assert graphed.materialize(build(x, y)) == expected


@dataclass(frozen=True)
class Declared:
    """A partitioned source over ``files`` copies of its rows that declares its own read list."""

    data: tuple[int, ...]
    files: int
    reads: ClassVar[list[object]] = []

    def __call__(self) -> list[int]:
        Declared.reads.append("all")
        return list(self.data) * self.files

    def projected_columns(self, outputs: Sequence[object]) -> list[str]:
        return ["declared"]

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition(f"toy://{i}", "", 0, len(self.data)) for i in range(self.files))

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> list[int]:
        Declared.reads.append(columns)
        return list(self.data)


@pytest.mark.parametrize(("files", "reads"), [(1, [("declared",)]), (2, ["all"])])
def test_materialize_reads_a_declared_list_unless_the_backend_cannot_concatenate(
    files: int, reads: list[object]
) -> None:
    Declared.reads.clear()
    s = Session(SumBackend())  # no concat: two partitions fall back to the lazy loader
    x = s.source("x", form=_form(), data=Declared((1, 2), files))
    assert graphed.materialize(s.record_op("add", [x, x])) == [2, 4] * files
    assert Declared.reads == reads


def test_materialize_evaluates_a_sessions_shared_nodes_once() -> None:
    s = Session(SumBackend())
    x = s.source("x", form=_form(), data=Loader((1, 2)))
    shared = s.record_op("add", [x, x])
    graphed.materialize(
        [s.record_op("add", [shared, x]), {"deeper": (s.record_op("add", [shared, shared]),)}]
    )
    assert SumBackend.calls.count("add") == 3
    assert Loader.loads == [(1, 2)]


# ---- plans over several Sessions are collated -------------------------------------------------
def _fold(outputs: Sequence[Any]) -> Any:
    return graphed.aggregate_plan(
        *outputs, reduce=lambda vs: [sum(v) for v in vs], combine=lambda a, b: [*a, *b], empty=list
    )


def test_aggregate_plan_over_several_sessions_names_collate() -> None:
    x = Session(SumBackend()).source("x", form=_form(), data=Rows((1, 2)))
    y = Session(SumBackend()).source("y", form=_form(), data=Rows((3,)))
    with pytest.raises(TypeError, match=r"one session; collate one plan per Session \(graphed.collate\)"):
        _fold([x, y])
    plan = graphed.collate({"x": _fold([x]), "y": _fold([y])})
    assert SequentialRunner().run(plan).value == {"x": [3], "y": [3]}
