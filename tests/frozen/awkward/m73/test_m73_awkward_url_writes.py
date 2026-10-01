"""m73: the awkward writers (`parquet_write`, `to_parquet`, `to_parquet(select=)`) write their parts
to an fsspec URL."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

import graphed
import graphed.awkward as ga
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core import Partition
from graphed.core.execution import SequentialRunner

EVENTS = ak.Array(
    {
        "x": np.array([0.5, 1.5, 2.5, 3.5, 0.25, 4.5], dtype=np.float64),
        "pt": [[1.0], [], [2.0, 3.0], [4.0], [5.0, 0.5], []],
    }
)
EXTRA = r"pip install 'graphed\[checkpoint\]'"


class MemorySource:
    """A `PartitionedSource` over `EVENTS`, blind by step; reads without fsspec."""

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("mem://m73", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        return chunk_of(partition)


def chunk_of(partition: Partition) -> ak.Array:
    part = partition.resolve(len(EVENTS))
    return EVENTS[part.entry_start : part.entry_stop]


def partitioned() -> Array:
    session = Session(AwkwardBackend())
    tracer = ak.Array(EVENTS.layout.to_typetracer(forget_length=True))
    return session.source("events", form=AwkwardForm(tracer), data=MemorySource(), uri="mem://m73")


def in_memory() -> Any:
    return from_awkward(Session(AwkwardBackend()), "ev", EVENTS)


def by_step(p: Partition) -> str:
    return f"s{p.blind_step}/part.parquet"


def paths_only(values: list[Any]) -> list[str]:
    return [v for v in values if isinstance(v, str)]


def add(a: list[str], b: list[str]) -> list[str]:
    return a + b


def no_paths() -> list[str]:
    return []


def run_writes(*writes: Any) -> list[str]:
    plan = graphed.aggregate_plan(reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2, writes=list(writes))
    paths: list[str] = SequentialRunner().run(plan).value
    return paths


def halves() -> list[ak.Array]:
    return [chunk_of(Partition.blind("mem://m73", "", s, 2)) for s in range(2)]


def varied_inputs(ev: Any) -> tuple[Any, Any]:
    return gak.zip({"x": ev.x, "pt": ev.pt}, depth_limit=1), graphed.vary(ev.x > 1, "cut", up=ev.x > 2, down=ev.x > 0)


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory: a URL write that creates a local tree creates it here."""
    here = tmp_path / "cwd"
    here.mkdir()
    monkeypatch.chdir(here)
    return here


@pytest.fixture
def url_root(cwd: Path) -> Iterator[str]:
    fsspec = pytest.importorskip("fsspec")
    name = f"m73-{uuid.uuid4().hex}"
    yield f"memory://{name}"
    fs = fsspec.filesystem("memory")
    if fs.exists(f"/{name}"):
        fs.rm(f"/{name}", recursive=True)


def on_url_fs(url_root: str) -> list[str]:
    import fsspec  # noqa: PLC0415

    return sorted("memory:/" + p for p in fsspec.filesystem("memory").find(url_root.removeprefix("memory:/")))


def fetch(url: str, local: Path) -> str:
    """The part's bytes, copied to a local file for readers that take a local path."""
    import fsspec  # noqa: PLC0415

    with fsspec.open(url, "rb") as handle:
        local.write_bytes(handle.read())
    return str(local)


def block_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [k for k in sys.modules if k == "fsspec" or k.startswith("fsspec.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "fsspec", None)


def test_parquet_write_lands_each_part_on_the_url_filesystem(url_root: str, cwd: Path, tmp_path: Path) -> None:
    ev = partitioned()
    rec = gak.zip({"x": ev.x * 2.0, "pt": ev.pt}, depth_limit=1)
    destination = f"{url_root}/pw"
    paths = run_writes(ga.parquet_write(rec, destination, name=by_step, metadata={"kind": "MC"}))
    expected = [f"{destination}/s{s}/part.parquet" for s in range(2)]
    assert paths == expected
    assert on_url_fs(url_root) == expected
    for step, (path, chunk) in enumerate(zip(paths, halves(), strict=True)):
        oracle = str(tmp_path / f"oracle_{step}.parquet")
        eager = ak.zip({"x": chunk.x * 2.0, "pt": chunk.pt}, depth_limit=1)
        pq.write_table(ak.to_arrow_table(eager).replace_schema_metadata({"kind": "MC"}), oracle)
        written = pq.read_table(fetch(path, tmp_path / f"got_{step}.parquet"))
        assert written.schema.metadata == {b"kind": b"MC"}
        assert written.equals(pq.read_table(oracle))
    assert os.listdir(cwd) == []


def test_parquet_write_only_a_url_destination_needs_fsspec(
    tmp_path: Path, cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev = partitioned()
    block_fsspec(monkeypatch)
    local = str(tmp_path / "local")
    paths = run_writes(ga.parquet_write(ev.x * 2.0, local, name=by_step))
    assert paths == [os.path.join(local, f"s{s}/part.parquet") for s in range(2)]
    for path, chunk in zip(paths, halves(), strict=True):
        assert pq.read_table(path)["data"].to_pylist() == (chunk.x * 2.0).to_list()
    with pytest.raises(ImportError, match=EXTRA):
        run_writes(ga.parquet_write(ev.x * 2.0, f"memory://m73-{uuid.uuid4().hex}/pw", name=by_step))
    assert os.listdir(cwd) == []


def test_to_parquet_writes_its_parts_on_the_url_filesystem(url_root: str, cwd: Path) -> None:
    import fsspec  # noqa: PLC0415

    ev = in_memory()
    destination = f"{url_root}/tp"
    paths = ga.to_parquet(gak.zip({"x": ev.x * 2.0, "pt": ev.pt}, depth_limit=1), destination, steps_per_file=2)
    expected = [f"{destination}/part-{i:05d}.parquet" for i in range(2)]
    assert paths == expected
    assert on_url_fs(url_root) == expected
    for path, chunk in zip(expected, [EVENTS[:3], EVENTS[3:]], strict=True):
        with fsspec.open(path, "rb") as handle:
            back = ak.from_arrow(pq.read_table(handle))
        assert back.to_list() == ak.zip({"x": chunk.x * 2.0, "pt": chunk.pt}, depth_limit=1).to_list()
    assert os.listdir(cwd) == []


def test_varied_to_parquet_writes_its_part_on_the_url_filesystem(url_root: str, cwd: Path, tmp_path: Path) -> None:
    ev = in_memory()
    record, mask = varied_inputs(ev)
    destination = f"{url_root}/var"
    paths = ga.to_parquet(record, destination, select=mask)
    assert paths == [f"{destination}/part-00000.parquet"]
    assert on_url_fs(url_root) == paths
    (oracle,) = ga.to_parquet(record, str(tmp_path / "oracle"), select=mask)
    got = ga.read_varied(fetch(paths[0], tmp_path / "got.parquet"))
    want = ga.read_varied(oracle)
    assert list(got) == list(want) == ["nominal", "cut_up", "cut_down"]
    assert {k: v.to_list() for k, v in got.items()} == {k: v.to_list() for k, v in want.items()}
    assert os.listdir(cwd) == []


@pytest.mark.parametrize("varied", [False, True], ids=["plain", "select"])
def test_to_parquet_to_a_url_without_fsspec_names_the_extra(
    varied: bool, cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev = in_memory()
    record, mask = varied_inputs(ev)
    destination = f"memory://m73-{uuid.uuid4().hex}/tp"
    block_fsspec(monkeypatch)
    with pytest.raises(ImportError, match=EXTRA):
        if varied:
            ga.to_parquet(record, destination, select=mask)
        else:
            ga.to_parquet(record, destination, steps_per_file=2)
    assert os.listdir(cwd) == []
