"""A Session recording several datasets writes the one an array reads."""

from __future__ import annotations

import os

import numpy as np
import pytest

from graphed import Session

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

import graphed.numpy.io as gio
from graphed.numpy import NumpyBackend


def test_to_parquet_writes_the_source_its_array_reads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = {}
    for name, xs in (("events", [1.0, 2.0, 3.0]), ("other", [9.0])):
        paths[name] = os.path.join(tmp_path, f"{name}.parquet")
        pq.write_table(pa.table({"x": xs}), paths[name])
    s = Session(NumpyBackend())
    gio.from_parquet(s, "other", paths["other"])
    g = gio.from_parquet(s, "events", paths["events"])
    out = gio.to_parquet(g["x"] * 2.0, os.path.join(tmp_path, "out"))
    assert isinstance(out, list)
    np.testing.assert_array_equal(
        np.concatenate([pq.read_table(p)["data"].to_numpy() for p in out]), [2.0, 4.0, 6.0]
    )
