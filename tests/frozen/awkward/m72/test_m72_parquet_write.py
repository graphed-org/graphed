"""m72 (c): `parquet_write` reproduces an existing format's parts exactly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from m72_awkward_fixtures import (  # noqa: E402
    EVENTS,
    add,
    by_step,
    dump_to_parquet,
    no_paths,
    option_record,
    paths_only,
    resolved_chunk,
    source,
    write_input,
)

import graphed  # noqa: E402
import graphed.awkward as ga  # noqa: E402
import graphed.awkward.io  # noqa: E402
from graphed import GraphedError  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core.execution import SequentialRunner  # noqa: E402


def _run(*writes: Any, steps: int = 2) -> list[str]:
    plan = graphed.aggregate_plan(
        reduce=paths_only, combine=add, empty=no_paths, steps_per_file=steps, writes=list(writes)
    )
    tasks = sorted(plan.tasks, key=lambda t: t.key)
    paths: list[str] = SequentialRunner().run(plan).value
    assert len(paths) == len(writes) * len(tasks)
    return paths


def test_parquet_write_reproduces_a_dump_to_parquet_part(tmp_path: Path) -> None:
    assert ga.parquet_write is graphed.awkward.io.parquet_write
    ev = source(write_input(tmp_path / "in.parquet"), steps=2)
    rec = option_record(ev, gak.zip, gak.mask)
    write = ga.parquet_write(
        rec[sorted(rec.fields)], str(tmp_path / "out"), name=by_step,
        metadata={"sum_w": gak.sum(ev.w), "kind": "MC"}, arrow_options={"extensionarray": False},
    )
    assert isinstance(write, graphed.write.PartWrite)
    plan = graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2, writes=[write])
    paths = SequentialRunner().run(plan).value
    for task, path in zip(sorted(plan.tasks, key=lambda t: t.key), paths, strict=True):
        chunk = resolved_chunk(EVENTS, task.partition)
        chunk_sum = np.sum(chunk.w.to_numpy())
        assert str(chunk_sum) != str(float(chunk_sum))
        eager = option_record(chunk, ak.zip, ak.mask)
        assert str(eager.type).startswith(f"{len(chunk)} * ?{{")
        oracle = str(tmp_path / f"oracle_{task.partition.blind_step}.parquet")
        dump_to_parquet(eager, oracle, {"sum_w": str(chunk_sum), "kind": "MC"})
        schema = pq.read_schema(path)
        assert schema.names == ["jag", "nest", "num"]
        assert schema.metadata == {b"sum_w": str(chunk_sum).encode(), b"kind": b"MC"}
        assert schema.equals(pq.read_schema(oracle), check_metadata=True)
        assert pq.read_table(path).equals(pq.read_table(oracle))


def test_parquet_write_defaults_are_the_libraries_defaults(tmp_path: Path) -> None:
    ev = source(write_input(tmp_path / "in.parquet"), steps=2)
    plain = gak.zip({"a": ev.w, "b": ev.n}, depth_limit=1)
    paths = _run(
        ga.parquet_write(plain, str(tmp_path / "rec"), name=by_step),
        ga.parquet_write(ev.w * 2, str(tmp_path / "flat"), name=by_step),
        ga.parquet_write(ev.w * 2, str(tmp_path / "col"), name=by_step, column="val"),
    )
    rec_paths = sorted(p for p in paths if f"{os.sep}rec{os.sep}" in p)
    flat_paths = sorted(p for p in paths if f"{os.sep}flat{os.sep}" in p)
    col_paths = sorted(p for p in paths if f"{os.sep}col{os.sep}" in p)
    halves = [EVENTS[:5], EVENTS[5:]]
    for step, chunk in enumerate(halves):
        eager = ak.zip({"a": chunk.w, "b": chunk.n}, depth_limit=1)
        oracle = str(tmp_path / f"oracle_rec_{step}.parquet")
        pq.write_table(ak.to_arrow_table(eager), oracle)
        schema = pq.read_schema(rec_paths[step])
        assert schema.equals(pq.read_schema(oracle), check_metadata=True)
        assert schema.metadata == ak.to_arrow_table(eager).schema.metadata
        assert [schema.field(n).nullable for n in ("a", "b")] == [False, False]
        assert pq.read_table(rec_paths[step]).equals(pq.read_table(oracle))
        compression = pq.ParquetFile(rec_paths[step]).metadata.row_group(0).column(0).compression
        assert compression == pq.ParquetFile(oracle).metadata.row_group(0).column(0).compression

        wrapped = str(tmp_path / f"oracle_flat_{step}.parquet")
        pq.write_table(ak.to_arrow_table(ak.Array({"data": chunk.w * 2})), wrapped)
        assert pq.read_schema(flat_paths[step]).names == ["data"]
        assert pq.read_schema(flat_paths[step]).equals(pq.read_schema(wrapped), check_metadata=True)
        assert pq.read_table(flat_paths[step]).equals(pq.read_table(wrapped))
        assert pq.read_schema(col_paths[step]).names == ["val"]
        assert ak.from_parquet(col_paths[step]).val.to_list() == (chunk.w * 2).to_list()


def test_parquet_write_refuses_a_varied(tmp_path: Path) -> None:
    assert ga.VERB_DISPOSITIONS["parquet_write"] == "refusing"
    ev = source(write_input(tmp_path / "in.parquet"))
    varied = graphed.vary(ev.w, "jes", up=ev.w * 1.5, down=ev.w * 0.5)
    with pytest.raises(GraphedError, match="does not accept a Varied"):
        ga.parquet_write(varied, str(tmp_path / "v"), name=by_step)
    with pytest.raises(GraphedError, match="does not accept a Varied"):
        ga.parquet_write(ev.w, str(tmp_path / "v"), name=by_step, metadata={"s": varied})
    assert isinstance(ga.parquet_write(ev.w, str(tmp_path / "v"), name=by_step, metadata={"s": "x"}), graphed.write.PartWrite)
    assert not (tmp_path / "v").exists()
