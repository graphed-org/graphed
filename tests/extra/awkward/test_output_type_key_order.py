"""The run-time ``output_type=`` check compares types as awkward does, keys in any order (#236)."""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.errors import StageError

L, N, R = ak.types.ListType, ak.types.NumpyType, ak.types.RecordType


def _run(data: ak.Array, declared: Any) -> list[Any]:
    session = Session(AwkwardBackend())
    x = from_awkward("ev", ak.Array({"v": data}), session=session).v
    return list(
        ak.to_list(session.materialize(graphed.apply(lambda a: a, x, output_type=declared)))
    )


def _list(rows: list[Any], leaf: dict[str, Any] | None = None) -> ak.Array:
    flat = np.asarray([v for row in rows for v in row], dtype="float64")
    offsets = ak.index.Index64(np.cumsum([0, *map(len, rows)]))
    content = ak.contents.NumpyArray(flat, parameters=leaf)
    return ak.Array(ak.contents.ListOffsetArray(offsets, content, parameters={"a": 1, "b": 2}))


RECORD = ak.zip({"x": ak.Array([1, 2]), "y": ak.Array([0.5, 1.5])})


@pytest.mark.parametrize(
    ("data", "declared"),
    [
        (_list([[0.5, 0.6], [0.8]]), L(N("float64"), parameters={"b": 2, "a": 1})),
        (
            _list([[0.5], [0.8]], leaf={"u": 1, "v": 2}),
            L(N("float64", parameters={"v": 2, "u": 1}), parameters={"b": 2, "a": 1}),
        ),
        (RECORD, R([N("float64"), N("int64")], ["y", "x"])),
        (_list([[], []]), L(N("float64"), parameters={"b": 2, "a": 1})),
    ],
    ids=["list-parameters", "leaf-parameters", "record-fields", "unknown-inside"],
)
def test_the_same_type_in_another_key_order_passes(data: ak.Array, declared: Any) -> None:
    assert _run(data, declared) == ak.to_list(data)


@pytest.mark.parametrize(
    ("data", "declared"),
    [
        (_list([[0.5]]), L(N("float64"), parameters={"a": 1})),
        (_list([[0.5]]), L(N("float64"), parameters={"b": 3, "a": 1})),
        (RECORD, R([N("float64"), N("int64")], ["y", "z"])),
    ],
    ids=["a-parameter-dropped", "a-parameter-differs", "a-field-differs"],
)
def test_a_different_type_is_still_refused(data: ak.Array, declared: Any) -> None:
    with pytest.raises(StageError, match="declares output_type"):
        _run(data, declared)
