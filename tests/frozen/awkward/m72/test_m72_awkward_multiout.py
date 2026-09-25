"""m72 (a)/(b) on the awkward backend: MC and data graphs in one plan, `num(axis=0)` as a
reduction, and the read projection of a writes-only plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from m72_awkward_fixtures import (  # noqa: E402
    EVENTS,
    RecordingSource,
    add,
    add_tuples,
    by_step,
    data_counts,
    data_zero,
    first_int,
    json_codec,
    mc_counts,
    mc_zero,
    no_paths,
    part_bytes,
    paths_only,
    recorded,
    source,
    write_input,
    zero,
)

import graphed  # noqa: E402
import graphed.awkward as ga  # noqa: E402
from graphed import GraphedError, compile_ir  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import GraphStore  # noqa: E402
from graphed.core.execution import SequentialRunner  # noqa: E402


def _inputs(tmp_path: Path) -> tuple[list[str], list[str]]:
    mc = [write_input(tmp_path / "mc_a.parquet"), write_input(tmp_path / "mc_b.parquet", EVENTS[3:])]
    data_events = ak.Array({"n": EVENTS.n, "pt": EVENTS.pt})
    data = [write_input(tmp_path / "data_a.parquet", data_events), write_input(tmp_path / "data_b.parquet", data_events[:7])]
    return mc, data


def _mc(paths: list[str], dest: Path) -> Any:
    ev = source(paths, steps=2)
    sel = ev[ev.n > 1]
    rec = gak.zip({"n": sel.n, "w": sel.w, "pt": sel.pt}, depth_limit=1)
    write = ga.parquet_write(rec, str(dest / "mc"), name=by_step, metadata={"sum_w": gak.sum(ev.w)})
    return graphed.aggregate_plan(
        gak.sum(ev.w), gak.sum(ev.n > 1), reduce=mc_counts, combine=add_tuples, empty=mc_zero,
        steps_per_file=2, writes=[write],
    )


def _data(paths: list[str], dest: Path) -> Any:
    ev = source(paths, steps=2)
    sel = ev[ev.n > 1]
    write = ga.parquet_write(
        gak.zip({"n": sel.n, "pt": sel.pt}, depth_limit=1), str(dest / "data"), name=by_step,
        metadata={"sum_w": "Data"},
    )
    return graphed.aggregate_plan(
        gak.sum(ev.n > 1), reduce=data_counts, combine=add_tuples, empty=data_zero, steps_per_file=2, writes=[write]
    )


def test_mc_and_data_graphs_collate_into_one_plan(tmp_path: Path) -> None:
    mc, data = _inputs(tmp_path)
    alone = {
        "mc": SequentialRunner().run(_mc(mc, tmp_path / "alone")).value,
        "data": SequentialRunner().run(_data(data, tmp_path / "alone")).value,
    }
    assert alone["mc"][0] > 0 and alone["data"][0] > 0
    runs = []
    for dest in ("one", "two"):
        collated = graphed.collate({"mc": _mc(mc, tmp_path / dest), "data": _data(data, tmp_path / dest)})
        assert len(collated.tasks) == 8
        assert SequentialRunner().run(collated).value == alone
        runs.append(part_bytes(tmp_path / dest))
    separate = part_bytes(tmp_path / "alone")
    assert len(separate) == 8
    assert runs[0] == separate
    assert runs[1] == runs[0]
    assert {pq.read_schema(tmp_path / "one" / p).metadata[b"sum_w"] for p in runs[0] if p.startswith("data")} == {b"Data"}
    mc_kv = {pq.read_schema(tmp_path / "one" / p).metadata[b"sum_w"] for p in runs[0] if p.startswith("mc")}
    assert len([p for p in runs[0] if p.startswith("mc")]) == 4 and b"Data" not in mc_kv


def test_num_axis0_is_a_reduction(tmp_path: Path) -> None:
    ev = source(write_input(tmp_path / "in.parquet"), steps=3)
    count = gak.num(ev.n, axis=0)
    store = GraphStore.deserialize(compile_ir(ev.session, count).ir)
    assert [store.nodes()[i]["kind"] for i in store.outputs()] == ["reduction"]
    with pytest.raises(GraphedError, match="feeds another node"):
        graphed.aggregate_plan(count * 2, reduce=first_int, combine=add, empty=zero, steps_per_file=3)
    plan = graphed.aggregate_plan(count, reduce=first_int, combine=add, empty=zero, steps_per_file=3)
    assert SequentialRunner().run(plan).value == len(EVENTS)
    with pytest.raises(GraphedError, match="a partitioned write has no combine step"):
        graphed.aggregate_plan(
            reduce=paths_only, combine=add, empty=no_paths, steps_per_file=3,
            writes=[ga.parquet_write(count, str(tmp_path / "out"), name=by_step)],
        )


def test_a_writes_only_plan_reads_only_the_written_columns(tmp_path: Path) -> None:
    data = ak.Array({"x": np.arange(6.0), "y": np.arange(10.0, 16.0), "z": np.arange(100.0, 106.0)})
    src = RecordingSource(data)
    ev = recorded(src)
    write = graphed.write.PartWrite(
        array=ev.x * 2, destination=str(tmp_path), name=by_step, codec=json_codec, metadata={"sy": gak.sum(ev.y)}
    )
    plan = graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2, writes=[write])
    paths = SequentialRunner().run(plan).value
    assert len(src.seen) == 2
    assert all(cols is not None and set(cols) == {"x", "y"} for cols in src.seen)
    docs = [json.loads(Path(p).read_text()) for p in paths]
    assert [d["values"] for d in docs] == [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]
    assert [d["kv"] for d in docs] == [{"sy": str(np.float64(33.0))}, {"sy": str(np.float64(42.0))}]
