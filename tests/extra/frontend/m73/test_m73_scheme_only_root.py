"""A scheme-only URL destination ("memory://") stays a URL: parts land on fsspec, never in the cwd."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import graphed
import graphed.write as gw
from graphed import Session
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.numpy import NumpyBackend, NumpyForm

FILES = {"fa": np.arange(0.0, 6.0), "fb": np.arange(20.0, 24.0)}


class FileSource:
    def __init__(self, files: Mapping[str, np.ndarray]) -> None:
        self.files = dict(files)

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(
            Partition.blind(u, "t", s, steps_per_file) for u in self.files for s in range(steps_per_file)
        )

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> np.ndarray:
        data = self.files[partition.uri]
        part = partition.resolve(len(data))
        return data[part.entry_start : part.entry_stop]


def url_json_codec(value: object, path: str, kv: Mapping[str, str] | None) -> None:
    from fsspec.core import url_to_fs  # noqa: PLC0415

    fs, stripped = url_to_fs(path)
    with fs.open(stripped, "w") as handle:
        json.dump(np.asarray(value).tolist(), handle)


@pytest.mark.parametrize("root", ["memory://", "memory:///"])
def test_join_part_keeps_a_scheme_only_root_a_url(root: str) -> None:
    assert gw.join_part(root, "n.json") == "memory:///n.json"


def test_scheme_only_destination_writes_on_fsspec_not_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsspec = pytest.importorskip("fsspec")
    monkeypatch.chdir(tmp_path)
    top = f"m73-{uuid.uuid4().hex}"
    x = Session(NumpyBackend()).source(
        "x", form=NumpyForm(np.dtype("float64"), shape=(None,)), data=FileSource(FILES)
    )
    write = gw.PartWrite(
        array=x,
        destination="memory://",
        name=lambda p: f"{top}/{p.uri}-{p.blind_step}.json",
        codec=url_json_codec,
    )
    plan = graphed.aggregate_plan(
        reduce=lambda vs: [v for v in vs if isinstance(v, str)],
        combine=lambda a, b: a + b,
        empty=list,
        writes=[write],
    )
    try:
        paths = SequentialRunner().run(plan).value
        assert sorted(paths) == [f"memory:///{top}/{u}-0.json" for u in FILES]
        assert list(tmp_path.iterdir()) == []
        for u in FILES:
            with fsspec.open(f"memory:///{top}/{u}-0.json", "r") as handle:
                assert json.load(handle) == FILES[u].tolist()
    finally:
        fs = fsspec.filesystem("memory")
        if fs.exists(f"/{top}"):
            fs.rm(f"/{top}", recursive=True)
