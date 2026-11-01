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

import graphed
import graphed.debug as gd
import graphed.numpy as gnp
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.awkward.io import from_parquet
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.errors import StageError, ValueStageError
from graphed.materialization import PlanPieces
from graphed.session import Session


@dataclass
class _Files:
    """A partitioned source recording the entries and columns of each read, clamping as a file."""

    chunks: tuple[ak.Array, ...]
    reads: list[tuple[int, int, tuple[str, ...]]] = field(default_factory=list)
    honours_ranges: bool = True

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return tuple(Partition.blind(f"f{i}", "t", 0, 1) for i in range(len(self.chunks)))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> Any:
        i = int(partition.uri[1:])
        chunk = self.chunks[i]
        if self.honours_ranges:
            part = partition.resolve(len(chunk))
            chunk = chunk[part.entry_start : part.entry_stop]
        self.reads.append((i, len(chunk), tuple(columns or chunk.fields)))
        return chunk

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must stay unused")


def _files(*sizes: int, honours_ranges: bool = True) -> tuple[Any, _Files]:
    start = 0
    chunks: list[ak.Array] = []
    for size in sizes:
        values: Any = np.arange(start, start + size, dtype=np.float64)
        chunks.append(ak.Array({"a": values, "b": values * 10, "c": values * 100}))
        start += size
    files = _Files(tuple(chunks), honours_ranges=honours_ranges)
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


def test_a_stage_error_over_a_value_error_stays_one_and_pickles() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "x", ak.Array({"a": [[1.0], [2.0, 3.0]], "b": [[1.0, 2.0], [3.0]]}))
    with pytest.raises(ValueError, match="StageError") as info:
        session.materialize(ev.a + ev.b)
    clone = pickle.loads(pickle.dumps(info.value))
    assert isinstance(clone, ValueStageError)
    assert clone == info.value


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
