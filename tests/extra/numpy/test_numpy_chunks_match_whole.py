"""Every numpy function: marked `reduction`, or its chunks are its whole value."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from graphed.aggregate import aggregate_plan
from graphed.array import Array
from graphed.core import GraphStore, Partition
from graphed.core.execution import SequentialRunner
from graphed.execute import _compile_cone
from graphed.numpy import VERB_DISPOSITIONS, NumpyBackend, NumpyForm, apply_gufunc, from_array
from graphed.numpy.array import _NUMPY_FUNCTIONS
from graphed.numpy.varied import SURFACE_DISPOSITIONS
from graphed.session import Session

_ONE = {
    "v": np.array([3.0, 1.0, 2.0]),
    "w": np.array([7.0, 7.0, 7.0]),
    "m": np.array([[3.0, 2.0], [1.0, 0.0], [5.0, 4.0]]),
    "c": np.array([[3.0], [1.0], [5.0]]),
    "i": np.array([1, 0, 1]),
    "b": np.array([True, False, True]),
}
_TWO = {
    "v": np.array([0.5, 7.0]),
    "w": np.array([3.0, 3.0]),
    "m": np.array([[6.0, 9.0], [8.0, 7.0]]),
    "c": np.array([[6.0], [8.0]]),
    "i": np.array([0, 1]),
    "b": np.array([False, True]),
}
_WHOLE = {key: np.concatenate([_ONE[key], _TWO[key]]) for key in _ONE}
_AXES: tuple[dict[str, Any], ...] = ({}, *({"axis": a} for a in (None, 0, 1, -1, -2)))
_RANGE = (0.0, 9.0)

#: `name: argument lists`, a string naming a column of the chunks
_PLAIN = dict.fromkeys(
    [
        "sum",
        "prod",
        "mean",
        "min",
        "amin",
        "max",
        "amax",
        "any",
        "all",
        "argmin",
        "argmax",
        "nansum",
        "nanprod",
        "nanmean",
        "nanmin",
        "nanmax",
        "nanargmin",
        "nanargmax",
        "std",
        "var",
        "nanstd",
        "nanvar",
        "cumsum",
        "cumprod",
        "nancumsum",
        "nancumprod",
        "ravel",
        "diff",
        "round",
        "around",
        "transpose",
    ],
    (("v",), ("m",)),
)
_ARGS: dict[str, tuple[tuple[Any, ...], ...]] = {
    **_PLAIN,
    "reshape": (("m", (-1,)), ("v", (-1, 1)), ("m", (-1, 2))),
    "squeeze": (("c",),),
    "swapaxes": (("m", 1, -1),),
    "expand_dims": (("v",), ("m",)),
    "clip": (("v", 1.0, 2.0),),
    "take": (("v", "i"), ("m", "i")),
    "astype": (("v", "f4"),),
    "where": (("b", "v", "v"), ("b", "v", 0.0)),
    "concatenate": ((["v", "v"],), (["m", "m"],)),
    "hstack": ((["v", "v"],), (["m", "m"],)),
    "isin": (("v", "w"),),
    "searchsorted": (("v", "v"),),
    "unique": (("v",),),
    "bincount": (("i",),),
    "histogram": (("v",),),
    "histogram2d": (("v", "v"),),
    "histogramdd": (("m",),),
}
_KWARGS: dict[str, dict[str, Any]] = {
    "histogram": {"bins": 2, "range": _RANGE},
    "histogram2d": {"bins": 2, "range": [_RANGE, _RANGE]},
    "histogramdd": {"bins": 2, "range": [_RANGE, _RANGE]},
}
_REFUSED = {"stack", "vstack"}


def _bound(value: Any, chunk: Any) -> Any:
    if isinstance(value, list):
        return [_bound(v, chunk) for v in value]
    return chunk[value] if isinstance(value, str) and value in _ONE else value


def _record(chunk: Any, call: Callable[[Any], Any]) -> tuple[Session, list[Array]]:
    session = Session(NumpyBackend())
    columns = {key: from_array(key, value, session=session) for key, value in chunk.items()}
    result = call(columns)
    arrays = result if isinstance(result, tuple) else (result,)
    return session, [a for a in arrays if isinstance(a, Array)]


def _marked(session: Session, arrays: list[Array]) -> bool:
    compiled = _compile_cone(session, *arrays)
    nodes = GraphStore.deserialize(bytes(compiled.ir)).nodes()
    return any(node["kind"] == "reduction" for node in nodes)


def _value(session: Session, arrays: list[Array]) -> list[Any]:
    return [np.asarray(session.materialize(array)) for array in arrays]


def _verdict(call: Callable[[Any], Any]) -> str:
    """Return ``marked``, ``matches``, ``no value``, or why this form is wrong."""
    try:
        session, arrays = _record(_WHOLE, call)
        whole = _value(session, arrays)
    except Exception:
        return "no value"
    if _marked(session, arrays):
        return "marked"
    try:
        parts = [_value(*_record(chunk, call)) for chunk in (_ONE, _TWO)]
        joined = [np.concatenate(column) for column in zip(*parts, strict=True)]
    except Exception as err:
        return f"chunks raise {type(err).__name__}: {str(err)[:80]}"
    if str(joined) != str(whole):
        return f"chunks {joined} != whole {whole}"
    return "matches"


def _call(name: str, args: tuple[Any, ...], axis: dict[str, Any]) -> Callable[[Any], Any]:
    def call(chunk: Any) -> Any:
        return getattr(np, name)(*_bound(list(args), chunk), **_KWARGS.get(name, {}), **axis)

    return call


def test_the_table_is_the_registry() -> None:
    assert set(_ARGS) | _REFUSED == set(_NUMPY_FUNCTIONS)


@pytest.mark.parametrize("name", sorted(_ARGS))
def test_a_function_is_marked_or_its_chunks_are_its_whole_value(name: str) -> None:
    verdicts = {
        f"{args} {axis}": _verdict(_call(name, args, axis))
        for args in _ARGS[name]
        for axis in _AXES
    }
    assert {"marked", "matches"} & set(verdicts.values()), verdicts
    assert not {k: v for k, v in verdicts.items() if v not in ("marked", "matches", "no value")}


def test_the_method_surface_and_verbs_are_the_same_functions() -> None:
    assert set(SURFACE_DISPOSITIONS) - {"__array_function__"} <= set(_NUMPY_FUNCTIONS)
    assert apply_gufunc.__name__ in VERB_DISPOSITIONS


def test_a_key_is_marked_only_when_it_picks_rows_by_position() -> None:
    session, (by_mask, by_position) = _record(_WHOLE, lambda c: (c["v"][c["b"]], c["v"][c["i"]]))
    assert not _marked(session, [by_mask])
    assert _marked(session, [by_position])


def test_a_reduction_of_two_row_sets_reaches_a_plans_reduce() -> None:
    @dataclass
    class Chunks:
        def partitions(self, target_bytes: int) -> tuple[Partition, ...]:
            return tuple(Partition.blind(uri, "t", 0, 1) for uri in ("0", "1"))

        def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
            return (_ONE["v"], _TWO["v"])[int(partition.uri)]

        def __call__(self) -> Any:
            return _WHOLE["v"]

    session = Session(NumpyBackend())
    v = session.source("chunks", form=NumpyForm(np.dtype("f8"), shape=(None,)), data=Chunks())
    found = np.isin(v, v[v > 2.0])
    plan = aggregate_plan(
        found,
        reduce=lambda values: [values[0].tolist()],
        combine=operator.add,
        empty=list,
        backend=lambda: session.backend,
    )
    assert SequentialRunner().run(plan).value == [[True, False, False], [False, True]]
