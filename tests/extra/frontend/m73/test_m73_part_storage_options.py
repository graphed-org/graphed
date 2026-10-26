"""A ``PartWrite``'s ``storage_options`` reach the filesystem its part directory is created on."""

from __future__ import annotations

import importlib
import json
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import graphed
import graphed.write as gw
from graphed import Session
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.numpy import NumpyBackend, NumpyForm

fsspec = pytest.importorskip("fsspec")
from fsspec.implementations.memory import MemoryFileSystem  # noqa: E402


class TokenMemory(MemoryFileSystem):  # type: ignore[misc]  # fsspec is untyped
    """The memory filesystem, refused without ``token=`` (as S3 with a custom endpoint would be)."""

    protocol = "tokmem"

    def __init__(self, *args: Any, token: str | None = None, **kwargs: Any) -> None:
        if token is None:
            raise PermissionError("tokmem needs token=")
        super().__init__(*args, **kwargs)

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        return str(MemoryFileSystem._strip_protocol("memory://" + path.removeprefix("tokmem://")))


class FileSource:
    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run inside a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return (Partition.blind("fa", "t", 0, 1),)

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> np.ndarray:
        return np.arange(3.0)


def token_json_codec(value: object, path: str, kv: Mapping[str, str] | None) -> None:
    fs, stripped = gw.url_fs(path, token="t")
    with fs.open(stripped, "w") as handle:
        json.dump(np.asarray(value).tolist(), handle)


def test_storage_options_reach_the_part_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        importlib.import_module("fsspec.registry")._registry, "tokmem", TokenMemory
    )
    top = f"m73-{uuid.uuid4().hex}"
    x = Session(NumpyBackend()).source(
        "x", form=NumpyForm(np.dtype("float64"), shape=(None,)), data=FileSource()
    )
    write = gw.PartWrite(
        array=x,
        destination=f"tokmem://{top}",
        name=lambda p: f"{p.uri}.json",
        codec=token_json_codec,
        storage_options={"token": "t"},
    )
    plan = graphed.aggregate_plan(
        reduce=lambda vs: [v for v in vs if isinstance(v, str)],
        combine=lambda a, b: a + b,
        empty=list,
        writes=[write],
    )
    fs = TokenMemory(token="t")
    try:
        assert SequentialRunner().run(plan).value == [f"tokmem://{top}/fa.json"]
        assert json.loads(fs.cat_file(f"/{top}/fa.json")) == [0.0, 1.0, 2.0]
    finally:
        if fs.exists(f"/{top}"):
            fs.rm(f"/{top}", recursive=True)
