"""m73: the format-agnostic write base and `aggregate_plan(writes=)` write parts to an fsspec URL.

Awkward-free (numpy backend, a toy codec), so the free-threaded job can collect it; the memory://
tests skip where fsspec is not installed."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import graphed
import graphed.parquet as gpq
import graphed.write as gw
from graphed import Array, Session
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.numpy import NumpyBackend, NumpyForm

FILES = {"fa": np.arange(0.0, 6.0), "fb": np.arange(20.0, 24.0)}
#: (path, parent directory existed on the part's own filesystem) per codec call in THIS process
CALLS: list[tuple[str, bool]] = []


class FileSource:
    """A `PartitionedSource` over in-memory "files" (uri -> 1-D array)."""

    def __init__(self, files: Mapping[str, np.ndarray]) -> None:
        self.files = dict(files)

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind(u, "t", s, steps_per_file) for u in self.files for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> np.ndarray:
        return chunk_of(partition)


def chunk_of(partition: Partition) -> np.ndarray:
    data = FILES[partition.uri]
    part = partition.resolve(len(data))
    return data[part.entry_start : part.entry_stop]


def recorded() -> Array:
    session = Session(NumpyBackend())
    return session.source("x", form=NumpyForm(np.dtype("float64"), shape=(None,)), data=FileSource(FILES))


def by_step(p: Partition) -> str:
    return f"{p.uri}/{p.blind_step}.json"


def paths_only(values: list[Any]) -> list[str]:
    return [v for v in values if isinstance(v, str)]


def add(a: list[str], b: list[str]) -> list[str]:
    return a + b


def no_paths() -> list[str]:
    return []


def url_json_codec(value: object, path: str, kv: Mapping[str, str] | None) -> None:
    """A third-party codec: it gets the part's string and writes through fsspec itself."""
    from fsspec.core import url_to_fs  # noqa: PLC0415

    fs, stripped = url_to_fs(path)
    CALLS.append((path, bool(fs.isdir(stripped.rsplit("/", 1)[0]))))
    with fs.open(stripped, "w") as handle:
        json.dump(np.asarray(value).tolist(), handle)


def local_json_codec(value: object, path: str, kv: Mapping[str, str] | None) -> None:
    CALLS.append((path, os.path.isdir(os.path.dirname(path))))
    with open(path, "w") as handle:
        json.dump(np.asarray(value).tolist(), handle)


def write(array: Array, destination: str, codec: Any = url_json_codec) -> gw.PartWrite:
    return gw.PartWrite(array=array, destination=destination, name=by_step, codec=codec)


def run(*writes: gw.PartWrite) -> tuple[list[list[str]], list[str]]:
    """(each task's `part_paths`, in key order; the paths the run reports)."""
    plan = graphed.aggregate_plan(
        reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2, writes=list(writes)
    )
    named = [list(plan.process.part_paths(t.partition)) for t in sorted(plan.tasks, key=lambda t: t.key)]
    return named, SequentialRunner().run(plan).value


def tasks_names() -> list[str]:
    return [f"{u}/{s}.json" for u in FILES for s in range(2)]


@pytest.fixture(autouse=True)
def _reset() -> None:
    CALLS.clear()


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


def read_url(url: str) -> Any:
    import fsspec  # noqa: PLC0415

    with fsspec.open(url, "r") as handle:
        return json.load(handle)


def block_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [k for k in sys.modules if k == "fsspec" or k.startswith("fsspec.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "fsspec", None)


def test_part_path_under_a_url_is_the_destination_slash_name() -> None:
    pytest.importorskip("fsspec")
    assert gw.part_path("memory://m73-a//", 7, suffix=".json") == "memory://m73-a/part-00007.json"
    assert gw.part_path("memory://m73-a/", 7, prefix="p", suffix=".x") == "memory://m73-a/p-00007.x"
    eos = "root://cmseos.fnal.gov//store/user/someone/skim"
    assert gw.part_path(eos, 3, suffix=".parquet") == f"{eos}/part-00003.parquet"
    assert gw.part_path(eos + "/", 3, suffix=".parquet") == f"{eos}/part-00003.parquet"
    assert gpq.part_path("memory://m73-a//", 1) == "memory://m73-a/part-00001.parquet"


def test_aggregate_write_lands_each_part_on_the_url_filesystem(url_root: str, cwd: Path) -> None:
    x = recorded()
    named, reported = run(write(x * 2.0, f"{url_root}/out//"))
    expected = [f"{url_root}/out/{n}" for n in tasks_names()]
    assert named == [[p] for p in expected]
    assert reported == expected
    assert [(p, True) for p in expected] == CALLS
    for p, (uri, step) in zip(expected, [(u, s) for u in FILES for s in range(2)], strict=True):
        assert read_url(p) == (chunk_of(Partition.blind(uri, "t", step, 2)) * 2.0).tolist()
    assert os.listdir(cwd) == []


def test_two_writes_of_one_url_part_are_refused_and_distinct_ones_run(url_root: str, cwd: Path) -> None:
    x = recorded()
    with pytest.raises(ValueError, match="same part"):
        graphed.aggregate_plan(
            reduce=paths_only, combine=add, empty=no_paths, steps_per_file=2,
            writes=[write(x, f"{url_root}/a"), write(x * 3.0, f"{url_root}/a/")],
        )
    assert CALLS == []
    _named, reported = run(write(x, f"{url_root}/a"), write(x * 3.0, f"{url_root}/b"))
    assert sorted(reported) == sorted(f"{url_root}/{d}/{n}" for d in "ab" for n in tasks_names())
    for n, (uri, step) in zip(tasks_names(), [(u, s) for u in FILES for s in range(2)], strict=True):
        chunk = chunk_of(Partition.blind(uri, "t", step, 2))
        assert read_url(f"{url_root}/a/{n}") == chunk.tolist()
        assert read_url(f"{url_root}/b/{n}") == (chunk * 3.0).tolist()
    assert os.listdir(cwd) == []


def test_only_a_url_destination_needs_fsspec(tmp_path: Path, cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block_fsspec(monkeypatch)
    x = recorded()
    local = str(tmp_path / "local")
    _named, reported = run(write(x * 2.0, local, codec=local_json_codec))
    assert reported == [os.path.join(local, n) for n in tasks_names()]
    for p in reported:
        with open(p) as handle:
            assert len(json.load(handle)) in (2, 3)
    CALLS.clear()
    with pytest.raises(ImportError, match=r"pip install 'graphed\[checkpoint\]'"):
        run(write(x * 2.0, f"memory://m73-{uuid.uuid4().hex}/out"))
    assert CALLS == []
    assert os.listdir(cwd) == []
