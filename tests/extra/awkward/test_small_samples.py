"""``graphed.head``, ``materialize(entries=/partition=)`` and the reads they cost."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

import graphed
import graphed.debug as gd
import graphed.numpy as gnp
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.awkward.io import from_parquet, read_parquet_partition
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.core.plan import _partition_bytes, _partition_from_json, _partition_json
from graphed.errors import PackKeyError, StageError
from graphed.materialization import PlanPieces
from graphed.parquet import row_groups_for
from graphed.session import Session


@dataclass
class _Files:
    """A partitioned source recording the entries and columns of each read, clamping as a file."""

    chunks: tuple[ak.Array, ...]
    reads: list[tuple[int, int, tuple[str, ...]]] = field(default_factory=list)
    honours_ranges: bool = True
    #: moves a blind range to multiples of this, which uproot's align_baskets does to baskets
    basket: int | None = None

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind(f"f{i}", "t", 0, 1) for i in range(len(self.chunks)))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
        i = int(partition.uri[1:])
        chunk = self.chunks[i]
        if self.honours_ranges:
            part = partition.resolve(len(chunk))
            start, stop = part.entry_start, part.entry_stop
            if self.basket and partition.is_blind and stop != len(chunk):
                start, stop = (round(e / self.basket) * self.basket for e in (start, stop))
            chunk = chunk[start:stop]
        self.reads.append((i, len(chunk), tuple(columns or chunk.fields)))
        return chunk

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must stay unused")


def _files(
    *sizes: int, honours_ranges: bool = True, basket: int | None = None
) -> tuple[Any, _Files]:
    start = 0
    chunks: list[ak.Array] = []
    for size in sizes:
        values: Any = np.arange(start, start + size, dtype=np.float64)
        chunks.append(ak.Array({"a": values, "b": values * 10, "c": values * 100}))
        start += size
    files = _Files(tuple(chunks), honours_ranges=honours_ranges, basket=basket)
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(chunks[0].layout.to_typetracer(forget_length=True)))
    return session.source("files", form=form, data=files), files


def _rows_read(files: _Files) -> int:
    return sum(rows for _, rows, _ in files.reads)


def test_head_reads_a_prefix_of_the_used_columns() -> None:
    ev, files = _files(1000)
    assert ak.to_list(graphed.head(ev.a * 2, 3)) == [0.0, 2.0, 4.0]
    assert files.reads == [(0, 3, ("a",))]


def test_head_widens_past_a_cut_that_empties_the_first_file() -> None:
    ev, files = _files(10, 10, 10)
    assert ak.to_list(graphed.head(ev.a[ev.a >= 25], 2)) == [25.0, 26.0]
    assert _rows_read(files) < 3 * 30


def test_head_returns_what_there_is_when_the_data_ends() -> None:
    ev, _ = _files(4, 4)
    assert ak.to_list(graphed.head(ev.a[ev.a >= 6], 5)) == [6.0, 7.0]


@pytest.mark.parametrize("entries", [5, 50, 70, 100])
def test_entries_is_exact_when_the_reader_moves_blind_ranges(entries: int) -> None:
    ev, files = _files(100, basket=40)
    assert ak.to_list(graphed.materialize(ev.a, entries=entries)) == list(range(entries))
    assert _rows_read(files) == entries


def test_head_is_exact_when_the_reader_moves_blind_ranges() -> None:
    ev, _ = _files(100, basket=40)
    assert ak.to_list(graphed.head(ev.a, 5)) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_cap_makes_a_blind_partitions_range_exact() -> None:
    blind = Partition.blind("f", "t", 0, 1)
    assert blind.is_blind
    assert not blind.first(3).is_blind


def test_head_of_nothing_reads_one_entry_and_returns_no_rows() -> None:
    ev, files = _files(4)
    assert ak.to_list(graphed.head(ev.a, 0)) == []
    assert _rows_read(files) == 1


def test_head_past_a_reduction_is_the_whole_value_not_the_prefixs() -> None:
    ev, files = _files(5, 5)
    centred = ev.a - gak.mean(ev.a, axis=None)
    assert ak.to_list(graphed.head(centred, 2)) == [-4.5, -3.5]
    assert sum(rows for _, rows, _ in files.reads) == 10


def test_head_refuses_a_negative_count_and_a_varied_array() -> None:
    ev, _ = _files(4)
    with pytest.raises(ValueError, match="at least 0"):
        graphed.head(ev.a, -1)
    varied = graphed.vary(ev.a, "scale", up=ev.a * 2, down=ev.a / 2)
    with pytest.raises(graphed.GraphedError, match=r"graphed\.head does not accept a Varied"):
        graphed.head(varied)


def test_gak_head_and_session_materialize_read_only_used_columns() -> None:
    ev, files = _files(100)
    x = ev.a + 1
    assert ak.to_list(gak.head(x, 2)) == [1.0, 2.0]
    assert files.reads == [(0, 2, ("a",))]
    files.reads.clear()
    assert ak.num(ev.session.materialize(x), axis=0) == 100
    assert files.reads == [(0, 100, ("a",))]


def test_head_on_an_in_memory_source_and_a_numpy_backend() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "x", ak.Array({"a": [1.0, 2.0, 3.0]}))
    assert ak.to_list(graphed.head(ev.a * 2, 2)) == [2.0, 4.0]
    flat = gnp.from_array("y", np.arange(6.0), session=Session(gnp.NumpyBackend()))
    assert graphed.head(flat * 2, 2).tolist() == [0.0, 2.0]


def test_head_and_entries_on_a_columns_mapping_source() -> None:
    session = Session(gnp.NumpyBackend())
    columns = {"a": np.arange(6.0), "b": np.arange(6.0) * 2}
    rec = gnp.from_record("r", session=session, **columns)
    assert np.asarray(graphed.head(rec.a, 2)).tolist() == [0.0, 1.0]
    assert np.asarray(graphed.materialize(rec.b, entries=3)).tolist() == [0.0, 2.0, 4.0]


def _parquet(tmp_path: Path, name: str, values: list[float]) -> str:
    path = tmp_path / name
    ak.to_parquet(ak.Array({"x": values, "y": values}), path)
    return str(path)


def test_entries_and_head_span_parquet_files(tmp_path: Path) -> None:
    paths = [_parquet(tmp_path, "a.parquet", [1.0, 2.0]), _parquet(tmp_path, "b.parquet", [3.0])]
    ev = from_parquet("ds", paths, session=Session(AwkwardBackend()))
    assert ak.to_list(graphed.materialize(ev.x, entries=3)) == [1.0, 2.0, 3.0]
    assert ak.to_list(graphed.head(ev.x[ev.x > 2], 1)) == [3.0]
    assert ak.to_list(graphed.materialize(ev.x, entries=10)) == [1.0, 2.0, 3.0]


def test_entries_counts_input_events_and_stops_at_the_files_it_needs() -> None:
    ev, files = _files(10, 10, 10)
    out = graphed.materialize(ev.a[ev.a % 2 == 0], entries=15)
    assert len(out) == 8
    assert [(i, rows) for i, rows, _ in files.reads] == [(0, 10), (1, 5)]


def test_entries_cuts_a_source_that_ignores_the_range() -> None:
    ev, files = _files(10, honours_ranges=False)
    assert len(graphed.materialize(ev.a, entries=4)) == 4
    assert _rows_read(files) == 10


def test_entries_treats_the_sample_as_the_dataset() -> None:
    ev, _ = _files(10)
    assert graphed.materialize(gak.sum(ev.a, axis=None), entries=4) == 6.0


def test_entries_keeps_a_nesting_across_sessions() -> None:
    one, _ = _files(10)
    two, _ = _files(20)
    out = graphed.materialize({"a": [one.a], "b": (two.b, 7)}, entries=2)
    assert ak.to_list(out["a"][0]) == [0.0, 1.0]
    assert ak.to_list(out["b"][0]) == [0.0, 10.0]
    assert out["b"][1] == 7


class _Total:
    def __init__(self, array: Any) -> None:
        self.array = array

    def __graphed_pieces__(self) -> PlanPieces:
        return PlanPieces(
            outputs=(self.array,),
            reduce=lambda values: float(ak.sum(values[0])),
            combine=lambda a, b: a + b,
            empty=lambda: 0.0,
            finalize=float,
            on_compiled=lambda _compiled: None,
        )


def test_entries_fills_a_deferred_object_from_the_sample() -> None:
    ev, files = _files(10, 10)
    assert graphed.materialize(_Total(ev.a), entries=3) == 3.0
    assert _rows_read(files) == 3


@pytest.mark.parametrize("entries", [0, -1, True, 1.5])
def test_entries_refuses_anything_but_a_positive_int(entries: Any) -> None:
    ev, _ = _files(4)
    with pytest.raises(ValueError, match="positive int"):
        graphed.materialize(ev.a, entries=entries)


def test_entries_and_partition_exclude_each_other() -> None:
    ev, files = _files(4)
    with pytest.raises(ValueError, match="not both"):
        graphed.materialize(ev.a, entries=1, partition=files.partitions(1)[0])


def test_a_plan_cannot_be_sampled() -> None:
    ev, _ = _files(4)
    plan = graphed.aggregate_plan(
        ev.a, reduce=lambda values: 0, combine=lambda a, b: a, empty=lambda: 0
    )
    with pytest.raises(TypeError, match="runs whole"):
        graphed.materialize(plan, entries=1)


def test_a_failure_in_a_sample_says_what_ran() -> None:
    ev, _ = _files(4)
    bad = ev.a.map(_explode_above(-1.0), name="bad", output_type="float64")
    with pytest.raises(StageError, match=r"partition first 2 entries"):
        graphed.materialize(bad, entries=2)


def _explode_above(limit: float) -> Any:
    def explode(values: Any) -> Any:
        if ak.max(values) > limit:
            raise RuntimeError("too large")
        return values

    return explode


def test_partition_runs_that_chunk() -> None:
    ev, files = _files(3, 3)
    out = graphed.materialize(ev.a, partition=files.partitions(1)[1])
    assert ak.to_list(out) == [3.0, 4.0, 5.0]
    assert [i for i, _, _ in files.reads] == [1]


def test_partition_reproduces_a_tasks_failure() -> None:
    ev, _ = _files(3, 3)
    bad = ev.a.map(_explode_above(4.0), name="bad", output_type="float64")
    plan = graphed.aggregate_plan(
        bad,
        reduce=lambda values: 0,
        combine=lambda a, b: a,
        empty=lambda: 0,
        backend=lambda: ev.session.backend,
    )
    with pytest.raises(StageError) as ran:
        SequentialRunner().run(plan)
    with pytest.raises(StageError) as again:
        graphed.materialize(bad, partition=plan.tasks[1].partition)
    assert again.value.partition == ran.value.partition == str(plan.tasks[1].partition)
    assert again.value.user_frame.lineno == ran.value.user_frame.lineno
    assert (again.value.cause_type, again.value.cause_message) == ("RuntimeError", "too large")


def test_partition_refuses_rows_that_are_not_one_row_set() -> None:
    ev, files = _files(3)
    short = ev.a[:2]
    with pytest.raises(StageError, match="different row sets"):
        graphed.materialize(ev.a + short, partition=files.partitions(1)[0])


def test_a_partition_no_source_holds_is_refused() -> None:
    ev, _ = _files(3)
    with pytest.raises(ValueError, match="no source of these results holds"):
        graphed.materialize(ev.a, partition=Partition("elsewhere", "t", 0, 1))


def test_gd_run_selects_a_real_partition() -> None:
    ev, files = _files(3, 3)
    out = gd.run(ev.session, ev.a, partition=files.partitions(1)[1])
    assert ak.to_list(out) == [3.0, 4.0, 5.0]


def test_gd_run_failure_names_the_partition_or_the_label() -> None:
    ev, files = _files(3)
    part = files.partitions(1)[0]
    bad = ev.a.map(_explode_above(-1.0), name="bad", output_type="float64")
    with pytest.raises(StageError) as chunk:
        gd.run(ev.session, bad, partition=part)
    assert chunk.value.partition == str(part)
    with pytest.raises(StageError) as named:
        gd.run(ev.session, bad, partition=part, label="the skim")
    assert named.value.partition == "the skim"


def test_gd_run_refuses_a_string_partition() -> None:
    ev, files = _files(3)
    with pytest.raises(TypeError, match=r"graphed\.core\.Partition.*label="):
        gd.run(ev.session, ev.a, partition="Events@0:5")  # type: ignore[arg-type]
    assert files.reads == []


def test_session_materialize_raises_an_attributed_error() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "x", ak.Array({"a": [[1.0], [2.0, 3.0]], "b": [[1.0, 2.0], [3.0]]}))
    bad = ev.a + ev.b
    with pytest.raises(StageError) as info:
        session.materialize(bad)
    assert info.value.user_frame.source.endswith("ev.a + ev.b")


def test_a_pack_key_overflow_is_a_located_value_error() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "x", ak.Array({"run": [1], "event": [2**20]}))
    key = graphed.pack_key(ev, on=("run", "event"))
    with pytest.raises(PackKeyError, match="pack_key: field 'event'") as info:
        session.materialize(key)
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, StageError)
    assert info.value.user_frame.source.startswith("graphed.pack_key")
    assert pickle.loads(pickle.dumps(info.value)) == info.value


def test_a_pack_key_overflow_in_a_plan_names_the_line_and_the_partition(tmp_path: Path) -> None:
    path = _parquet(tmp_path, "keys.parquet", [1.0])
    ak.to_parquet(ak.Array({"run": [1], "event": [2**20]}), path)
    ev = from_parquet("ds", path, session=Session(AwkwardBackend()))
    key = graphed.pack_key(ev, on=("run", "event"))
    plan = graphed.aggregate_plan(
        key,
        reduce=lambda values: 0,
        combine=lambda a, b: a,
        empty=lambda: 0,
        backend=lambda: ev.session.backend,
    )
    with pytest.raises(PackKeyError) as info:
        SequentialRunner().run(plan)
    assert info.value.user_frame.source.startswith("graphed.pack_key")
    assert info.value.partition == str(plan.tasks[0].partition)


def test_a_foreign_array_is_refused_by_session_materialize() -> None:
    ev, _ = _files(3)
    with pytest.raises(TypeError, match="different Session"):
        Session(AwkwardBackend()).materialize(ev.a)


@pytest.mark.parametrize(
    ("partition", "entries", "expected"),
    [
        (Partition.blind("f", "t", 0, 1), 3, (0, 3)),
        (Partition.blind("f", "t", 0, 1), 50, (0, 10)),
        (Partition.blind("f", "t", 1, 2), 2, (5, 7)),
        (Partition("f", "t", 4, 8), 3, (4, 7)),
        (Partition("f", "t", 4, 8), 30, (4, 8)),
    ],
)
def test_first_caps_a_partition_at_its_own_entries(
    partition: Partition, entries: int, expected: tuple[int, int]
) -> None:
    part = partition.first(entries).resolve(10)
    assert (part.entry_start, part.entry_stop) == expected


def test_a_second_cap_never_widens_a_blind_partition() -> None:
    capped = Partition.blind("f", "t", 0, 1).first(2)
    assert capped.first(5).resolve(10).entry_stop == 2
    assert capped.first(1).resolve(10).entry_stop == 1


def test_a_cap_survives_a_plans_json_and_keys_its_task() -> None:
    blind = Partition.blind("f", "t", 0, 1)
    capped = blind.first(3)
    assert _partition_from_json(_partition_json(capped)) == capped
    assert _partition_bytes(capped) != _partition_bytes(blind)
    assert "blind_limit" not in _partition_json(blind)


def test_head_refuses_a_row_set_a_prefix_would_broadcast() -> None:
    ev, _ = _files(20)
    misaligned = ev.a + ev.b[ev.b > 0]
    with pytest.raises(StageError, match="different row sets"):
        graphed.head(misaligned, 1)


def test_entries_keeps_a_keepdims_reduction_broadcasting() -> None:
    ev, _ = _files(5)
    shifted = ev.a + gak.sum(ev.a, axis=0, keepdims=True)
    assert ak.to_list(graphed.materialize(shifted, entries=3)) == [3.0, 4.0, 5.0]


class _NoConcat(AwkwardBackend):
    def __getattribute__(self, name: str) -> Any:
        if name == "concat":
            raise AttributeError(name)
        return super().__getattribute__(name)


@dataclass
class _LoadedFiles(_Files):
    def __call__(self) -> ak.Array:
        return ak.concatenate(self.chunks)


def test_a_backend_that_cannot_concatenate_still_reaches_the_later_files() -> None:
    files = _LoadedFiles((ak.Array({"a": [1.0]}), ak.Array({"a": [2.0, 3.0]})))
    session = Session(_NoConcat())
    form = AwkwardForm(ak.Array(files.chunks[0].layout.to_typetracer(forget_length=True)))
    ev = session.source("files", form=form, data=files)
    assert ak.to_list(graphed.head(ev.a, 2)) == [1.0, 2.0]
    assert ak.to_list(graphed.materialize(ev.a, entries=3)) == [1.0, 2.0, 3.0]


def test_head_widening_loads_a_lazy_source_once() -> None:
    values = ak.Array({"a": np.arange(100.0)})
    calls: list[int] = []

    def load() -> ak.Array:
        calls.append(1)
        return values

    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(values.layout.to_typetracer(forget_length=True)))
    ev = session.source("lazy", form=form, data=load)
    assert ak.to_list(graphed.head(ev.a[ev.a > 90], 2)) == [91.0, 92.0]
    assert len(calls) == 1


def test_head_refuses_a_count_that_is_not_an_int() -> None:
    ev, files = _files(4)
    with pytest.raises(TypeError):
        graphed.head(ev.a, 2.5)  # type: ignore[arg-type]
    assert files.reads == []


class _Unreadable(_Files):
    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
        raise OSError("disk gone")


def test_gd_run_attributes_a_failed_partition_read() -> None:
    _, files = _files(3)
    broken = _Unreadable(files.chunks)
    session = Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(files.chunks[0].layout.to_typetracer(forget_length=True)))
    ev = session.source("broken", form=form, data=broken)
    with pytest.raises(StageError) as info:
        gd.run(session, ev.a, partition=broken.partitions(1)[0])
    assert info.value.cause_type == "OSError"


def test_gd_run_refuses_a_partition_no_source_holds() -> None:
    ev, _ = _files(3)
    with pytest.raises(ValueError, match="no source of these results holds") as info:
        gd.run(ev.session, ev.a, partition=Partition("elsewhere", "t", 0, 1))
    assert not isinstance(info.value, StageError)


def _row_groups(tmp_path: Path) -> str:
    path = tmp_path / "groups.parquet"
    pq.write_table(pa.table({"x": np.arange(10.0)}), path, row_group_size=4)
    return str(path)


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [(0, 3, ([0], 0)), (3, 5, ([0, 1], 0)), (5, 7, ([1], 4)), (8, 99, ([2], 8)), (4, 4, ([], 0))],
)
def test_row_groups_for_covers_a_range(
    tmp_path: Path, start: int, stop: int, expected: tuple[list[int], int]
) -> None:
    assert row_groups_for(_row_groups(tmp_path), start, stop) == expected


def test_a_parquet_partition_reads_only_its_row_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _row_groups(tmp_path)
    asked: list[Any] = []
    read = ak.from_parquet

    def counting(*args: Any, **kwargs: Any) -> Any:
        asked.append(kwargs["row_groups"])
        return read(*args, **kwargs)

    monkeypatch.setattr(ak, "from_parquet", counting)
    chunk = read_parquet_partition(Partition(path, "", 5, 9))
    assert ak.to_list(chunk.x) == [5.0, 6.0, 7.0, 8.0]
    assert asked == [[1, 2]]
    assert len(read_parquet_partition(Partition(path, "", 4, 4))) == 0
    assert asked[-1] is None
