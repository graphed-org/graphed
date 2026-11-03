"""An uncaught StageError reads on its own: note, types, map name, raising line, chunk label."""

from __future__ import annotations

import json
import pickle
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.awkward as ga
import graphed.awkward.functions as gak
import graphed.debug as gd
from graphed.aggregate import _type_string, read_label
from graphed.core import Partition, Plan, SequentialRunner, chunk_label
from graphed.debug.report import RunReport
from graphed.errors import SourceFrame


def _calib(values: Any) -> Any:
    raise RuntimeError("calibration table missing")


def _sum(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _first(values: list[Any]) -> Any:
    return values[0]


def _source(tmp_path: Path, steps_per_file: int = 3) -> Any:
    path = tmp_path / "data.parquet"
    pytest.importorskip("pyarrow")
    ak.to_parquet(ak.Array({"x": [1.0, 2.0, 3.0] * 10, "n": [[1, 2], [], [3]] * 10}), path)
    session = graphed.Session(ga.AwkwardBackend())
    return ga.from_parquet("events", str(path), steps_per_file=steps_per_file, session=session)


def _failure(output: Any) -> gd.StageError:
    plan = graphed.aggregate_plan(output, reduce=_first, combine=_sum, empty=_zero)
    with pytest.raises(gd.StageError) as info:
        SequentialRunner().run(plan)
    return info.value


def test_uncaught_rendering_shows_the_source_and_arrow(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.n[:, 5], axis=None))
    text = "".join(traceback.format_exception(err))
    assert "events.n[:, 5]" in text
    assert "--> File" in text
    assert text.count("--> File") == 1


def test_note_survives_pickle_and_a_reraise_adds_none(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.n[:, 5], axis=None))
    loaded = pickle.loads(pickle.dumps(err))
    assert loaded.__notes__ == err.__notes__ == [err.view()]
    with pytest.raises(gd.StageError) as again:
        raise loaded
    assert len(again.value.__notes__) == 1


def test_equal_inputs_pickle_byte_identically() -> None:
    def build() -> gd.StageError:
        return gd.StageError(
            op="index",
            frames=(SourceFrame("a.py", 1, "f", "x[0]"),),
            input_forms=("3 * float64",),
            partition="f.root (entries 0-3)",
            cause_type="IndexError",
            cause_message="oob",
            opt_level=1,
            raised_at=SourceFrame("a.py", 2, "g", "raise"),
        )

    assert pickle.dumps(build()) == pickle.dumps(build())


def test_plan_path_reports_types_not_class_names(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.x.map(_calib, name="calib"), axis=None))
    assert err.input_forms == ("10 * float64",)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.zeros((2, 3), dtype="float32"), "2 * 3 * float32"),
        (np.float64(1.0), "float64"),
        ("not an array", "str"),
    ],
)
def test_type_string_without_awkward(value: object, expected: str) -> None:
    assert _type_string(value) == expected


def test_type_string_caps_a_long_record() -> None:
    record = ak.Array([{f"field_{i}": 1.0 for i in range(200)}])
    text = _type_string(record, limit=50)
    assert len(text) == 53 and text.endswith("...")


def test_map_failure_names_the_function_and_the_raising_line(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.x.map(_calib, name="calib"), axis=None))
    assert err.op == "map 'calib'"
    assert err.raised_at is not None
    assert (err.raised_at.filename, err.raised_at.function) == (__file__, "_calib")
    view = gd.format_traceback(err)
    assert f"raised in _calib(), {__file__}" in view
    assert 'raise RuntimeError("calibration table missing")' in view


def test_a_library_failure_has_no_user_raising_frame(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.n[:, 5], axis=None))
    assert err.raised_at is None
    assert "raised in" not in gd.format_traceback(err)


def test_blind_partition_shows_real_entries(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.x.map(_calib, name="calib"), axis=None))
    assert err.partition.startswith("chunk 1/3 of ")
    assert err.partition.endswith("data.parquet (entries 0-10)")
    assert "entry_stop" not in str(err)


def test_chunk_label_forms() -> None:
    blind = Partition.blind("f.root", "/Events;1", 6, 20)
    assert chunk_label(blind) == "chunk 7/20 of f.root:Events"
    assert chunk_label(blind, rows=2500) == "chunk 7/20 of f.root:Events (2500 entries)"
    assert (
        chunk_label(blind, file_entries=50000, rows=1)
        == "chunk 7/20 of f.root:Events (entries 15000-17500)"
    )
    assert chunk_label(Partition("f.root", "Events", 5, 9)) == "f.root:Events (entries 5-9)"
    assert chunk_label(Partition("f.parquet", "", 5, 9)) == "f.parquet (entries 5-9)"


class _Unknowing:
    def __init__(self, reader: Any, *, count: bool) -> None:
        self.reader = reader
        if count:
            self.num_entries = self._refuses

    def _refuses(self, partition: Partition) -> int:
        raise OSError("gone")

    def __getattr__(self, name: str) -> Any:
        if name == "num_entries":
            raise AttributeError(name)
        return getattr(self.reader, name)


@pytest.mark.parametrize("count", [False, True], ids=["no-hook", "hook-raises"])
def test_blind_label_falls_back_to_the_chunk_size(tmp_path: Path, count: bool) -> None:
    events = _source(tmp_path)
    output = gak.sum(events.x.map(_calib, name="calib"), axis=None)
    plan = graphed.aggregate_plan(output, reduce=_first, combine=_sum, empty=_zero)
    assert isinstance(plan, Plan)
    shipped: Any = plan.process
    process = replace(shipped, reader=_Unknowing(shipped.reader, count=count))
    with pytest.raises(gd.StageError) as info:
        process(plan.tasks[0].partition, graphed.core.execution.LocalResources())
    assert info.value.partition.endswith("data.parquet (10 entries)")


def test_an_unsized_chunk_does_not_mask_the_failure() -> None:
    blind = Partition.blind("f.root", "Events", 0, 2)
    assert read_label(object(), blind, np.array(1.0)) == "chunk 1/2 of f.root:Events"


def test_a_shuffle_map_failure_labels_the_chunk_like_a_plan(tmp_path: Path) -> None:
    events = _source(tmp_path)
    plan = graphed.shuffle_plan(
        graphed.repartition(events.x.map(_calib, name="calib"), n=1),
        reduce=_first,
        combine=_sum,
        empty=_zero,
    )
    with pytest.raises(gd.StageError) as info:
        SequentialRunner().run(plan)
    assert info.value.partition.endswith("data.parquet (entries 0-10)")


def test_the_debug_runner_names_the_raising_line(tmp_path: Path) -> None:
    events = _source(tmp_path)
    output = events.x.map(_calib, name="calib")
    with pytest.raises(gd.StageError) as info:
        gd.run(output.session, output)
    assert info.value.raised_at is not None
    assert info.value.raised_at.function == "_calib"


def test_worker_lost_error_is_a_picklable_stage_error_with_no_frame() -> None:
    err = graphed.WorkerLostError(partition="chunk 7/20 of f.root", worker="pid 4242")
    assert isinstance(err, gd.StageError)
    assert gd.WorkerLostError is graphed.WorkerLostError
    assert str(err).startswith("worker pid 4242 died running chunk 7/20 of f.root")
    assert err.frames == ()
    assert "no source line" in str(err.user_frame)
    assert "--> <no source line" in err.view()
    loaded = pickle.loads(pickle.dumps(err))
    assert type(loaded) is graphed.WorkerLostError
    assert loaded == err
    assert (loaded.worker, loaded.partition) == ("pid 4242", "chunk 7/20 of f.root")


def test_run_report_round_trips_the_raising_frame(tmp_path: Path) -> None:
    events = _source(tmp_path)
    err = _failure(gak.sum(events.x.map(_calib, name="calib"), axis=None))
    report = RunReport(
        outcome="failed",
        n_partitions=1,
        wall_s=0.0,
        tasks=(),
        failure=err,
        error=None,
        environment={},
        environment_digest="",
    )
    loaded = RunReport.from_json(json.loads(json.dumps(report.to_json())))
    assert loaded.failure == err


def test_run_report_round_trips_a_lost_worker() -> None:
    err = graphed.WorkerLostError(partition="chunk 1/1 of f.root", worker="w0")
    report = RunReport(
        outcome="failed",
        n_partitions=1,
        wall_s=0.0,
        tasks=(),
        failure=err,
        error=None,
        environment={},
        environment_digest="",
    )
    loaded = RunReport.from_json(json.loads(json.dumps(report.to_json())))
    assert type(loaded.failure) is graphed.WorkerLostError
    assert loaded.failure == err


def test_every_path_names_a_failing_map_and_its_chunk_alike(tmp_path: Path) -> None:
    events = _source(tmp_path)
    mapped = events.x.map(_calib, name="calib")
    output = gak.sum(mapped, axis=None)
    plan = graphed.aggregate_plan(
        output, reduce=_first, combine=_sum, empty=_zero, store=tmp_path / "cap"
    )
    assert isinstance(plan, Plan)
    with pytest.raises(gd.StageError) as ran:
        SequentialRunner().run(plan)
    partition = plan.tasks[0].partition
    seen = [ran.value]
    for again in (
        lambda: graphed.materialize(mapped, partition=partition),
        lambda: gd.run(events.session, mapped, partition=partition),
        lambda: list(gd.replay(plan, plan.tasks[0].key, output).steps()),
    ):
        with pytest.raises(gd.StageError) as info:
            again()
        seen.append(info.value)
    expected = ("map 'calib'", f"chunk 1/3 of {tmp_path / 'data.parquet'} (entries 0-10)")
    assert [(err.op, err.partition) for err in seen] == [expected] * 4
    with pytest.raises(gd.StageError) as traced:
        gd.trace(mapped, entries=2)
    assert traced.value.op == "map 'calib'"


def test_a_quoted_map_name_keeps_the_graph_source_well_formed(tmp_path: Path) -> None:
    events = _source(tmp_path)
    mapped = events.x.map(_calib, name="it's").map(lambda a: a)
    graph = gd.lower(events.session, mapped, opt_level=1)
    assert [op.op for op in graph.ops][-2:] == ['map "it\'s"', "map '<lambda>'"]
    mermaid = gd.visualize(graph, fmt="mermaid")
    assert "map #quot;it's#quot;" in mermaid
    assert "map '#lt;lambda#gt;'" in mermaid
    assert all(line.count('"') == 2 for line in mermaid.splitlines() if "[" in line)
    graphviz = gd.visualize(graph, fmt="graphviz")
    assert 'map \\"it\'s\\"' in graphviz
