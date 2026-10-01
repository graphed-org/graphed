"""m73: `graphed.numpy.io.to_parquet` writes its parts to an fsspec URL."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

import graphed.numpy as gn
import graphed.numpy.io as gio
from graphed import Array, Session
from graphed.core.execution import Plan, SequentialRunner
from graphed.numpy import NumpyBackend

DATA = np.arange(0.0, 9.0)


def recorded() -> Array:
    return gn.from_array(Session(NumpyBackend()), "x", DATA)


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


def block_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [k for k in sys.modules if k == "fsspec" or k.startswith("fsspec.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "fsspec", None)


def test_numpy_to_parquet_writes_its_parts_on_the_url_filesystem(url_root: str, cwd: Path) -> None:
    import fsspec  # noqa: PLC0415

    x = recorded()
    destination = f"{url_root}/np"
    expected = [f"{destination}/part-{i:05d}.parquet" for i in range(3)]
    plan = gio.to_parquet(x * 2.0, destination, steps_per_file=3, compute=False)
    assert isinstance(plan, Plan)
    tasks = sorted(plan.tasks, key=lambda t: t.key)
    assert [list(plan.process.part_paths(t.partition)) for t in tasks] == [[p] for p in expected]
    assert SequentialRunner().run(plan).value == expected
    on_fs = fsspec.filesystem("memory").find(url_root.removeprefix("memory:/"))
    assert sorted(on_fs) == [p.removeprefix("memory:/") for p in expected]
    back = []
    for p in expected:
        with fsspec.open(p, "rb") as handle:
            back.append(pq.read_table(handle)["data"].to_numpy())
    assert [len(b) for b in back] == [3, 3, 3]
    np.testing.assert_array_equal(np.concatenate(back), DATA * 2.0)
    assert os.listdir(cwd) == []


def test_numpy_only_a_url_destination_needs_fsspec(tmp_path: Path, cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block_fsspec(monkeypatch)
    x = recorded()
    local = str(tmp_path / "local")
    paths = gio.to_parquet(x * 2.0, local, steps_per_file=3)
    assert paths == [os.path.join(local, f"part-{i:05d}.parquet") for i in range(3)]
    np.testing.assert_array_equal(np.concatenate([pq.read_table(p)["data"].to_numpy() for p in paths]), DATA * 2.0)
    with pytest.raises(ImportError, match=r"pip install 'graphed\[checkpoint\]'"):
        gio.to_parquet(x * 2.0, f"memory://m73-{uuid.uuid4().hex}/np", steps_per_file=3)
    assert os.listdir(cwd) == []
