"""Every gak function, at each axis form: marked `reduction`, or its chunks are its whole value."""

from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward
from graphed.awkward import functions as gak
from graphed.awkward._dispositions import GAK_ARG_FIXTURES, GAK_DISPOSITIONS, GakSlot
from graphed.core import GraphStore, Partition
from graphed.core.execution import SequentialRunner
from graphed.errors import GraphedTypeError, StageError
from graphed.execute import _compile_cone
from graphed.session import Session

_ONE = ak.Array(
    {
        "x": [[3.0, 2.0], [], [1.0]],
        "y": [[1.0, 2.0], [], [3.0]],
        "s": [3.0, 1.0, 1.0],
        "t": [2.0, 1.0, 5.0],
        "o": [1, None, 3],
        "b": [True, False, True],
        "m": [[True, False], [], [True]],
        "r": [{"pt": [1.0, 2.0]}, {"pt": []}, {"pt": [3.0]}],
    }
)
_TWO = ak.Array(
    {
        "x": [[6.0], [5.0, 4.0, 7.0]],
        "y": [[9.0], [1.0, 2.0, 3.0]],
        "s": [1.0, 0.5],
        "t": [0.25, 7.0],
        "o": [None, 6],
        "b": [False, True],
        "m": [[False], [True, True, False]],
        "r": [{"pt": [4.0]}, {"pt": [5.0, 6.0, 7.0]}],
    }
)
_NESTED = {"flat": "s", "jagged": "x", "record": "r", "mask": "m", "option": "o"}
_ROWS = {**_NESTED, "jagged": "s", "mask": "b"}
_AXES: tuple[Any, ...] = (None, 0, 1, -1, -2)
_CASES = sorted(
    name
    for name, kind in GAK_DISPOSITIONS.items()
    if kind not in ("eager-metadata", "refusing")
    and name in GAK_ARG_FIXTURES
    and name not in ("apply_correction", "onnx_inference")  # Externals, row-wise by contract
)


def _bound(value: Any, events: Any, slots: dict[str, str]) -> Any:
    if isinstance(value, GakSlot):
        return events[slots[value.kind]]
    if isinstance(value, list | tuple):
        return [_bound(v, events, slots) for v in value]
    if isinstance(value, dict):
        return {k: _bound(v, events, slots) for k, v in value.items()}
    return value


def _forms(name: str) -> Iterator[tuple[dict[str, str], dict[str, Any]]]:
    takes_axis = "axis" in inspect.signature(getattr(gak, name)).parameters
    for slots in (_NESTED, _ROWS):
        yield slots, {}
        if takes_axis:
            for axis in _AXES:
                yield slots, {"axis": axis}


def _record(data: ak.Array, name: str, slots: dict[str, str], axis: dict[str, Any]) -> Any:
    session = Session(AwkwardBackend())
    events = from_awkward("m", data, session=session)
    fixture = GAK_ARG_FIXTURES[name]
    args = _bound(fixture.args, events, slots)
    kwargs = {**_bound(dict(fixture.kwargs), events, slots), **axis}
    result = getattr(gak, name)(*args, **kwargs)
    return session, list(result) if isinstance(result, tuple) else [result]


def _marked(session: Session, arrays: list[Any]) -> bool:
    compiled = _compile_cone(session, *arrays)
    nodes = GraphStore.deserialize(bytes(compiled.ir)).nodes()
    return any(node["kind"] == "reduction" for node in nodes)


def _value(session: Session, arrays: list[Any]) -> list[Any]:
    return [session.materialize(array) for array in arrays]


def _text(values: list[Any]) -> str:
    return str([ak.to_list(value) for value in values])


def _verdict(name: str, slots: dict[str, str], axis: dict[str, Any]) -> str:
    """Return ``marked``, ``matches``, ``no value``, or why this form is wrong."""
    try:
        session, arrays = _record(ak.concatenate([_ONE, _TWO]), name, slots, axis)
        whole = _value(session, arrays)
    except Exception:
        return "no value"
    if _marked(session, arrays):
        return "marked"
    try:
        parts = [_value(*_record(chunk, name, slots, axis)) for chunk in (_ONE, _TWO)]
        joined = [ak.concatenate(list(column)) for column in zip(*parts, strict=True)]
    except Exception as err:
        return f"chunks raise {type(err).__name__}: {str(err)[:80]}"
    if _text(joined) != _text(whole):
        return f"chunks {_text(joined)[:60]} != whole {_text(whole)[:60]}"
    return "matches"


@pytest.mark.parametrize("name", _CASES)
def test_a_function_is_marked_or_its_chunks_are_its_whole_value(name: str) -> None:
    verdicts = {
        f"{'rows' if slots is _ROWS else 'nested'} {axis}": _verdict(name, slots, axis)
        for slots, axis in _forms(name)
    }
    assert {"marked", "matches"} & set(verdicts.values()), verdicts
    assert not {k: v for k, v in verdicts.items() if v not in ("marked", "matches", "no value")}


@dataclass
class _Chunks:
    def partitions(self, target_bytes: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind(uri, "t", 0, 1) for uri in ("0", "1"))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        return (_ONE, _TWO)[int(partition.uri)]

    def __call__(self) -> ak.Array:
        return ak.concatenate([_ONE, _TWO])


def _events() -> Any:
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(_ONE.layout.to_typetracer(forget_length=True)))
    return session.source("chunks", form=form, data=_Chunks())


def _chunked(array: Any) -> list[Any]:
    plan = graphed.aggregate_plan(
        array,
        reduce=lambda values: [values[0]],
        combine=operator.add,
        empty=list,
        backend=lambda: array.session.backend,
    )
    chunks: list[Any] = ak.to_list(ak.concatenate(SequentialRunner().run(plan).value))
    return chunks


@pytest.mark.parametrize(
    ("select", "marked"),
    [
        (lambda e: e.s[e.s > 1], False),
        (lambda e: e.x[e.x > 1], False),
        (lambda e: e.s[e.o > 1], False),
        (lambda e: e.s.filter(e.s > 1), False),
        (lambda e: e.x[gak.argsort(e.x, axis=1)], False),
        (lambda e: e.s[gak.values_astype(e.s, "int64")], True),
        (lambda e: e.s.filter(gak.values_astype(e.s, "int64")), True),
    ],
)
def test_a_key_is_marked_only_when_it_picks_rows_by_position(
    select: Callable[[Any], Any], marked: bool
) -> None:
    selected = select(_events())
    assert _marked(selected.session, [selected]) is marked


@pytest.mark.parametrize("opt_level", [0, 1])
def test_two_row_sets_in_one_chunk_refuse_rather_than_broadcast(opt_level: int) -> None:
    events = _events()
    outlier = events.s - events.s[events.s == 3.0]
    plan = graphed.aggregate_plan(
        outlier,
        reduce=lambda values: [values[0]],
        combine=operator.add,
        empty=list,
        backend=lambda: outlier.session.backend,
        opt_level=opt_level,
    )
    with pytest.raises(StageError, match=r"\[1, 3\] rows in one chunk"):
        SequentialRunner().run(plan)


def test_a_whole_evaluation_still_broadcasts_a_length_one_row_set() -> None:
    events = _events()
    outlier = events.s - events.s[events.s == 3.0]
    assert ak.to_list(outlier.session.materialize(outlier)) == [0.0, -2.0, -2.0, -2.0, -2.5]


@pytest.mark.parametrize(
    "build",
    [
        lambda e: e.s[(e.s > 0.5) & (e.t > 1)] + e.t[e.s > 0.5][e.t[e.s > 0.5] > 1],
        lambda e: gak.unflatten(gak.flatten(e.x), gak.num(e.x)),
        lambda e: gak.unflatten(e.s, 1),
        lambda e: e.x[gak.argsort(e.x, axis=1)],
    ],
)
def test_aligned_rows_run_chunked_to_their_whole_value(build: Callable[[Any], Any]) -> None:
    array = build(_events())
    whole = build(_events())
    assert not _marked(array.session, [array])
    assert _chunked(array) == ak.to_list(whole.session.materialize(whole))


def test_grouping_rows_across_a_chunk_edge_is_marked() -> None:
    grouped = gak.unflatten(_events().s, 2)
    assert _marked(grouped.session, [grouped])


@pytest.mark.parametrize(
    ("function", "axis", "marked"),
    [
        (gak.local_index, -1, False),
        (gak.local_index, -2, True),
        (lambda r, axis: gak.pad_none(r, 3, axis=axis), -1, False),
        (lambda r, axis: gak.pad_none(r, 3, axis=axis), -2, True),
    ],
)
def test_a_negative_axis_of_a_record_counts_from_its_fields(
    function: Callable[..., Any], axis: int, marked: bool
) -> None:
    recorded = function(_events().r, axis=axis)
    assert _marked(recorded.session, [recorded]) is marked


def test_a_negative_axis_reaching_a_flat_field_of_a_record_is_marked() -> None:
    session = Session(AwkwardBackend())
    mixed = from_awkward("m", ak.Array({"s": [1.0], "x": [[1.0]]}), session=session)
    assert _marked(session, [gak.local_index(mixed, axis=-1)])


def test_a_scalar_record_is_refused_at_the_users_line() -> None:
    events = _events()
    with pytest.raises(GraphedTypeError, match=r"ak\.sort"):
        gak.sort(events[0], axis=-1)
    with pytest.raises(GraphedTypeError, match="getitem"):
        events.s[events[0]]


def test_a_mask_with_parameters_is_still_a_mask() -> None:
    session = Session(AwkwardBackend())
    data = ak.Array({"s": [1.0, 2.0], "b": ak.with_parameter(ak.Array([True, False]), "k", 1)})
    events = from_awkward("m", data, session=session)
    selected = events.s[events.b]
    assert not _marked(session, [selected])


def test_a_reduction_as_a_plans_output_reaches_reduce_with_unequal_inputs() -> None:
    events = _events()
    both = gak.concatenate([events.s, events.s[events.s > 1.0]], axis=0)
    assert _chunked(both) == [3.0, 1.0, 1.0, 3.0, 1.0, 0.5]


def _dests(array: Any) -> Any:
    plan = graphed.shuffle_plan(
        array,
        reduce=lambda values: [ak.to_list(values[0])],
        combine=operator.add,
        empty=list,
        backend=AwkwardBackend,
    )
    return SequentialRunner().run(plan).value


def test_two_row_sets_after_an_exchange_refuse_rather_than_broadcast() -> None:
    moved = graphed.repartition(_events(), n=1)
    with pytest.raises(StageError, match=r"\[1, 5\] rows in one chunk"):
        _dests(moved.s - moved.s[moved.s == 3.0])


def test_aligned_rows_after_an_exchange_run_to_their_whole_value() -> None:
    moved = graphed.repartition(_events(), n=1)
    assert sorted(_dests(moved.s[moved.s > 0.75] + 1.0)[0]) == [2.0, 2.0, 2.0, 4.0]
