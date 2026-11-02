"""An op pairing rows of two partitioned sources is refused; ``graphed.join`` matches them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gak
from graphed.core import SequentialRunner
from graphed.core.execution import LocalResources
from graphed.errors import GraphedTypeError, SourceRowsError, StageError
from graphed.shuffle import pick, split

WIDE = 2**40


def _files(root: Path, name: str, chunks: list[ak.Array]) -> list[str]:
    paths = [str(root / f"{name}{i}.parquet") for i in range(len(chunks))]
    for chunk, path in zip(chunks, paths, strict=True):
        ak.to_parquet(chunk, path)
    return paths


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, Any]:
    session = graphed.Session(AwkwardBackend())
    read: dict[str, Any] = {"session": session}
    tables = {
        "dy": ak.Array({"x": [1.0, 2.0, 3.0], "w": [1.0, 2.0, 3.0], "j": [[1.0], [], [2.0, 3.0]]}),
        "tt": ak.Array({"x": [10.0, 20.0, 30.0], "w": [0.5, 0.5, 0.5]}),
        "one": ak.Array({"x": [100.0]}),
    }
    for name, table in tables.items():
        read[name] = from_parquet(name, _files(tmp_path, name, [table]), session=session)
    return read


PAIRINGS: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda dy, tt: dy.x + tt.x,
    "with_field": lambda dy, tt: gak.with_field(dy, tt.x, "y"),
    "mask": lambda dy, tt: tt.x[dy.x > 1.0],
    "where": lambda dy, tt: gak.where(dy.x > 1.0, tt.x, 0.0),
    "zip": lambda dy, tt: gak.zip({"a": dy.x, "b": tt.x}),
    "weight": lambda dy, tt: gak.mean(dy.x, weight=tt.w),
    "inner_concatenate": lambda dy, tt: gak.concatenate(
        [gak.singletons(dy.x), gak.singletons(tt.x)], axis=1
    ),
    "external": lambda dy, tt: graphed.apply(lambda a, b: a + b, dy.x, tt.x),
    "varied": lambda dy, tt: dy.x + graphed.vary(tt.x, "sig", up=tt.x * 2.0),
    "filtered": lambda dy, tt: dy[dy.x > 1.0].x + tt.x,
    "joined_with_a_side": lambda dy, tt: (
        graphed.join(dy[["x"]], tt[["w"]], how="positional").x + dy.x
    ),
    "appended_in_other_order": lambda dy, tt: (
        gak.concatenate([dy.x, tt.x]) * gak.concatenate([tt.w, dy.w])
    ),
    "joined_in_other_order": lambda dy, tt: (
        graphed.join(dy[["x"]], tt[["w"]], how="positional").x
        + graphed.join(tt[["w"]], dy[["x"]], how="positional").w
    ),
}


@pytest.mark.parametrize("pair", PAIRINGS.values(), ids=PAIRINGS)
def test_pairing_two_sources_is_refused_at_the_line(
    sources: dict[str, Any], pair: Callable[[Any, Any], Any]
) -> None:
    with pytest.raises(SourceRowsError) as info:
        pair(sources["dy"], sources["tt"])
    text = str(info.value)
    assert "'dy' and 'tt'" in text
    assert "graphed.join(left, right, on=[...])" in text
    assert "how='positional'" in text
    assert "graphed.collate" in text
    assert info.value.provenance.filename == __file__


def test_the_refusal_is_a_located_error_and_a_type_error(sources: dict[str, Any]) -> None:
    with pytest.raises(SourceRowsError, match="one partitioned source") as info:
        sources["dy"].x + sources["tt"].x
    assert isinstance(info.value, GraphedTypeError)
    assert isinstance(info.value, TypeError)
    assert info.value.op == "add"


def test_a_one_event_source_does_not_broadcast(sources: dict[str, Any]) -> None:
    with pytest.raises(SourceRowsError, match="'dy' and 'one'"):
        sources["dy"].x + sources["one"].x


def test_scalars_cross_sources(sources: dict[str, Any]) -> None:
    dy, tt = sources["dy"], sources["tt"]
    norm = dy.w * gak.sum(tt.w)
    assert ak.to_list(graphed.materialize(dy.x * norm)) == [1.5, 6.0, 13.5]
    assert graphed.materialize(gak.sum(dy.x) + gak.sum(tt.x)) == 66.0


def test_a_record_element_carries_no_rows(sources: dict[str, Any]) -> None:
    dy, tt = sources["dy"], sources["tt"]
    assert graphed.materialize(dy[0].x + tt[0].x) == 11.0


def test_row_appending_pairs_nothing(sources: dict[str, Any]) -> None:
    dy, tt = sources["dy"], sources["tt"]
    xs = gak.concatenate([dy.x, tt.x])
    ws = gak.concatenate([dy.w, tt.w])
    assert ak.to_list(graphed.materialize(xs * ws)) == [1.0, 4.0, 9.0, 5.0, 10.0, 15.0]


def test_crossing_rows_pairs_nothing(sources: dict[str, Any]) -> None:
    dy, tt = sources["dy"], sources["tt"]
    crossed = gak.cartesian({"a": dy.x, "b": tt.x}, axis=0)
    assert len(graphed.materialize(crossed)) == 9


def test_in_memory_sources_stay_pairable(sources: dict[str, Any]) -> None:
    session = sources["session"]
    a = from_awkward("a", ak.Array({"x": [1.0, 2.0, 3.0]}), session=session)
    b = from_awkward("b", ak.Array({"x": [1.0]}), session=session)
    assert ak.to_list(graphed.materialize(a.x + b.x)) == [2.0, 3.0, 4.0]
    assert ak.to_list(graphed.materialize(sources["dy"].x + a.x)) == [2.0, 4.0, 6.0]


@pytest.fixture
def friends(tmp_path: Path) -> tuple[Any, Any]:
    idx = np.arange(12)
    events = ak.Array({"x": idx * 1.0, "j": ak.unflatten(idx * 1.0, np.ones(12, dtype=np.int64))})
    friend = ak.Array({"y": idx * 2.0})
    session = graphed.Session(AwkwardBackend())
    e = from_parquet("e", _files(tmp_path, "e", [events[:6], events[6:]]), session=session)
    f = from_parquet("f", _files(tmp_path, "f", [friend[:6], friend[6:]]), session=session)
    return e, f


def _map_payloads(plan: Any) -> list[bytes]:
    stage = plan.stages[0]
    process = stage.process.resolve()
    resources = LocalResources()
    return [process(task, (), resources) for task in stage.tasks]


def test_a_gather_picks_nothing_from_another_dests_position_task() -> None:
    assert split(pick({1: b"block"}, 1)) == {1: b"block"}
    assert split(pick({1: b"block"}, 0)) == {}


def test_positional_join_pairs_rows_with_no_key(friends: tuple[Any, Any]) -> None:
    joined = graphed.join(*friends, how="positional")
    assert ak.to_list(graphed.materialize(joined.x + joined.y)) == [i * 3.0 for i in range(12)]
    assert ak.to_list(graphed.materialize(gak.sum(joined.j, axis=1)))[:2] == [0.0, 1.0]
    assert "__joinkey__" not in joined.fields


def test_positional_join_plan_pairs_block_by_block(friends: tuple[Any, Any]) -> None:
    joined = graphed.join(*friends, how="positional")
    plan = graphed.join_plan(joined.x + joined.y, steps_per_file=3)
    assert [(st.kind, len(st.tasks)) for st in plan.stages] == [
        ("map_write", 6),
        ("map_write", 6),
        ("gather_join", 6),
    ]
    assert plan.stages[0].routing["scheme"] == "position"
    assert all(len(split(payload)) == 1 for payload in _map_payloads(plan))
    value = ak.concatenate(SequentialRunner().run(plan).value)
    assert ak.to_list(value) == [i * 3.0 for i in range(12)]


def test_two_joins_of_one_pair_are_two_row_sets(tmp_path: Path) -> None:
    left, right = _keyed(tmp_path)
    on = ["run", "event"]
    inner = graphed.join(left, right, on=on)
    outer = graphed.join(left, right, on=on, how="left")
    with pytest.raises(SourceRowsError, match="'left' and 'right'"):
        inner.x + outer.y
    assert ak.to_list(graphed.materialize(inner.x + inner.x))[0] == 2.0


def test_a_join_is_not_the_rows_it_read(friends: tuple[Any, Any]) -> None:
    e, f = friends
    appended = gak.concatenate([e.x, f.y])
    joined = graphed.join(e[["x"]], f[["y"]], how="positional")
    with pytest.raises(SourceRowsError, match="'e' and 'f'"):
        appended + joined.y


def test_a_join_of_a_table_takes_a_row_set_of_its_own(friends: tuple[Any, Any]) -> None:
    e, _ = friends
    table = from_awkward("table", ak.Array({"z": np.arange(12.0)}), session=e.session)
    joined = graphed.join(e[["x"]], table, how="positional")
    with pytest.raises(SourceRowsError):
        joined.z + e.x


def test_tables_join_with_no_row_set(friends: tuple[Any, Any]) -> None:
    session = friends[0].session
    a = from_awkward("a", ak.Array({"p": [1.0, 2.0]}), session=session)
    b = from_awkward("b", ak.Array({"q": [3.0, 4.0]}), session=session)
    joined = graphed.join(a, b, how="positional")
    assert ak.to_list(graphed.materialize(joined.p + a.p)) == [2.0, 4.0]
    other = graphed.join(
        a, from_awkward("c", ak.Array({"r": [5.0, 6.0]}), session=session), how="positional"
    )
    assert ak.to_list(graphed.materialize(joined.p + other.r)) == [6.0, 8.0]


def test_a_positional_join_of_selected_fields_reads_only_those(friends: tuple[Any, Any]) -> None:
    e, f = friends
    joined = graphed.join(e[["x"]], f[["y"]], how="positional")
    stages = graphed.join_plan(joined.x + joined.y).stages
    read: list[Any] = [stages[i].process.resolve() for i in (0, 1)]
    assert [r.columns for r in read] == [("x",), ("y",)]


def test_positional_join_refuses_unequal_rows_rather_than_broadcast(
    sources: dict[str, Any],
) -> None:
    one = gak.zip({"o": sources["one"].x})
    with pytest.raises(StageError, match="3 rows against 1"):
        graphed.materialize(graphed.join(sources["dy"], one, how="positional"))


def test_positional_join_plan_refuses_unequal_partitions(
    friends: tuple[Any, Any], tmp_path: Path
) -> None:
    e, _ = friends
    short = ak.Array({"y": [1.0]})
    s = from_parquet("short", _files(tmp_path, "short", [short]), session=e.session)
    with pytest.raises(TypeError, match="'e' has 4 and 'short' 2"):
        graphed.join_plan(graphed.join(e, s, how="positional"), steps_per_file=2)


def test_positional_join_plan_refuses_a_short_block(
    friends: tuple[Any, Any], tmp_path: Path
) -> None:
    e, _ = friends
    ys = ak.Array({"y": np.arange(5.0)})
    s = from_parquet("short", _files(tmp_path, "short", [ys, ys]), session=e.session)
    plan = graphed.join_plan(graphed.join(e, s, how="positional"), steps_per_file=1)
    with pytest.raises(StageError, match="6 rows against 5"):
        SequentialRunner().run(plan)


@pytest.mark.parametrize(
    "kwargs", [{}, {"on": ["x"], "how": "positional"}], ids=["neither", "both"]
)
def test_join_takes_a_key_or_positional(friends: tuple[Any, Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="or how='positional'"):
        graphed.join(*friends, **kwargs)


def test_positional_join_needs_records_with_distinct_fields(friends: tuple[Any, Any]) -> None:
    e, f = friends
    with pytest.raises(GraphedTypeError, match="two records"):
        graphed.join(e.x, f, how="positional")
    with pytest.raises(GraphedTypeError, match="select fields on one side"):
        graphed.join(e, e, how="positional")


def test_gak_join_is_positional_too_but_not_grouped(friends: tuple[Any, Any]) -> None:
    e, f = friends
    assert ak.to_list(graphed.materialize(gak.join(e, f, how="positional").y))[:2] == [0.0, 2.0]
    with pytest.raises(TypeError, match="grouped=True"):
        gak.join(e, f, how="positional", grouped=True)
    with pytest.raises(TypeError, match="grouped=True"):
        gak.join(e, f, grouped=True)


def _keyed(tmp_path: Path) -> tuple[Any, Any]:
    wide = np.array([WIDE, WIDE + 2**20, 5, 2**20], dtype=np.uint64)
    left = ak.Array({"run": [1, 1, 1, 1], "event": wide, "x": [1.0, 2.0, 3.0, 4.0]})
    right = ak.Array({"run": [1, 1, 1], "event": wide[[3, 0, 1]], "y": [40.0, 10.0, 20.0]})
    session = graphed.Session(AwkwardBackend())
    return (
        from_parquet("left", _files(tmp_path, "left", [left]), session=session),
        from_parquet("right", _files(tmp_path, "right", [right]), session=session),
    )


def test_a_keyed_join_matches_event_numbers_too_wide_to_pack(tmp_path: Path) -> None:
    left, right = _keyed(tmp_path)
    joined = graphed.join(left, right, on=["run", "event"])
    assert ak.to_list(graphed.materialize(joined.y / joined.x)) == [10.0, 10.0, 10.0]
    assert ak.to_list(graphed.materialize(gak.join(left, right, on=["run", "event"]).y)) == [
        10.0,
        20.0,
        40.0,
    ]


def test_a_keyed_join_plan_matches_event_numbers_too_wide_to_pack(tmp_path: Path) -> None:
    left, right = _keyed(tmp_path)
    joined = graphed.join(left, right, on=["run", "event"])
    value = SequentialRunner().run(graphed.join_plan(joined.y / joined.x)).value
    assert ak.to_list(ak.concatenate(value)) == [10.0, 10.0, 10.0]


def test_a_grouped_join_matches_event_numbers_too_wide_to_pack(tmp_path: Path) -> None:
    left, right = _keyed(tmp_path)
    grouped = gak.join(left, right, on=["run", "event"], grouped=True)
    assert ak.to_list(graphed.materialize(grouped.y)) == [[10.0], [20.0], [40.0]]


def test_public_pack_key_still_refuses_a_wide_value(tmp_path: Path) -> None:
    left, _ = _keyed(tmp_path)
    with pytest.raises(StageError, match=r"2\*\*20"):
        graphed.materialize(graphed.pack_key(left, on=["run", "event"]))
