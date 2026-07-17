"""Implementer regression (M15 iteration 2): the parquet path must work WITHOUT pandas.

CI caught what the shared dev venv masked: pyarrow's ChunkedArray.to_numpy and
DataType.to_pandas_dtype both route through pyarrow's pandas SHIM, importing pandas — which is
not a dependency of this package. This test blocks pandas imports outright and exercises schema
translation, materialize, partition reads, and writes."""

from __future__ import annotations

import builtins
import os
import sys

import numpy as np
import pytest

import graphed.numpy.io as gio
from graphed import Session
from graphed import parquet as gpq
from graphed.numpy import NumpyBackend

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


@pytest.fixture
def no_pandas(monkeypatch):  # type: ignore[no-untyped-def]
    real_import = builtins.__import__

    def block(name: str, *args: object, **kwargs: object) -> object:
        if name == "pandas" or name.startswith("pandas."):
            raise ModuleNotFoundError("No module named 'pandas' (blocked by test)")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", block)
    monkeypatch.delitem(sys.modules, "pandas", raising=False)


def test_full_parquet_path_without_pandas(no_pandas, tmp_path) -> None:  # type: ignore[no-untyped-def]
    pq.write_table(
        pa.table({"x": [1.0, 2.0, 3.0], "n": [1, 2, 3], "f": [True, False, True]}),
        os.path.join(tmp_path, "a.parquet"),
    )
    s = Session(NumpyBackend())
    g = gio.from_parquet(s, "events", str(tmp_path))  # schema translation
    np.testing.assert_array_equal(np.asarray(s.materialize(g["x"] + g["n"])), [2.0, 4.0, 6.0])
    part = gpq.make_partitions([os.path.join(tmp_path, "a.parquet")], steps_per_file=2, open_files=False)[0]
    chunk = gio.read_parquet_partition(part)
    assert set(chunk) == {"x", "n", "f"}
    paths = gio.to_parquet(g["x"] * 2.0, os.path.join(tmp_path, "out"), steps_per_file=2)
    assert len(paths) == 2
