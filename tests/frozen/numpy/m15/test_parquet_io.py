"""M15.3: partitioned parquet I/O for the numpy backend — RECTILINEAR ONLY (parity plan).

The numpy specialization of the `graphed.parquet` base handles flat columnar data: every parquet
column must be a fixed-width primitive (numeric/bool). Jagged or nested columns are REFUSED at
construction with an error naming the offending column and pointing at graphed-awkward — numpy
bags cannot represent them honestly. Reads are wired from the M5 field-touch projection; writes
follow the R15.4 disabled/enabled consistency contract via the compiled IR.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from graphed import Session

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: I001
import pyarrow.parquet as pq

from graphed import parquet as gpq
from graphed.core.execution import Plan, SequentialRunner

import graphed.numpy as gn
import graphed.numpy.io as gio
from graphed.numpy import NumpyBackend, from_record

LENGTHS = [6, 4, 5]


def _cols(n: int, offset: int = 0) -> dict[str, list[object]]:
    return {
        "x": [float(offset + i) for i in range(n)],
        "n": [offset + i for i in range(n)],
        "flag": [bool((offset + i) % 2) for i in range(n)],
    }


@pytest.fixture
def dataset(tmp_path):  # type: ignore[no-untyped-def]
    whole: dict[str, list[object]] = {"x": [], "n": [], "flag": []}
    offset = 0
    for i, n in enumerate(LENGTHS):
        cols = _cols(n, offset)
        pq.write_table(pa.table(cols), os.path.join(tmp_path, f"d-{i}.parquet"))
        for k in whole:
            whole[k] += cols[k]
        offset += n
    return str(tmp_path), {k: np.asarray(v) for k, v in whole.items()}


def _s() -> Session:
    return Session(NumpyBackend())


# ---- deferred reading ------------------------------------------------------------------------
def test_from_parquet_records_a_rectilinear_record_source(dataset) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    g = gio.from_parquet(s, "events", where)
    form = s.form(g)
    assert dict(form.fields).keys() == {"x", "n", "flag"}  # type: ignore[attr-defined]
    out = s.materialize(g)
    for k, ref in whole.items():
        np.testing.assert_array_equal(np.asarray(out[k]), ref)  # type: ignore[index]


def test_field_access_and_projection_work_on_parquet_sources(dataset) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    g = gio.from_parquet(s, "events", where)
    expr = (g["x"] * 2.0 + g["n"]).reduce("sum")
    assert gn.project(expr).columns_for("events") == frozenset({"x", "n"})
    assert float(np.asarray(s.materialize(expr))) == pytest.approx(float((whole["x"] * 2 + whole["n"]).sum()))


def test_columns_filter(dataset) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    s = _s()
    g = gio.from_parquet(s, "events", where, columns=["x"])
    form = s.form(g)
    assert dict(form.fields).keys() == {"x"}  # type: ignore[attr-defined]


def test_jagged_columns_are_refused_by_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = os.path.join(tmp_path, "jagged.parquet")
    pq.write_table(pa.table({"ok": [1.0, 2.0], "jets": [[1.0, 2.0], [3.0]]}), p)
    s = _s()
    with pytest.raises(TypeError, match=r"jets.*graphed-awkward|graphed-awkward.*jets"):
        gio.from_parquet(s, "events", p)


def test_partition_reads_tile_the_dataset(dataset) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    paths = gpq.discover(where)
    for open_files in (True, False):
        parts = gpq.make_partitions(paths, steps_per_file=2, open_files=open_files)
        chunks = [gio.read_parquet_partition(p, columns=["x"]) for p in parts]
        got = np.concatenate([c["x"] for c in chunks])
        np.testing.assert_array_equal(got, whole["x"])
        assert all(set(c.keys()) == {"x"} for c in chunks)  # column projection honored


# ---- deferred writing ------------------------------------------------------------------------
def test_to_parquet_roundtrips_through_the_compiled_ir(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    g = gio.from_parquet(s, "events", where)
    expr = g["x"] * 2.0 + g["n"]
    outdir = os.path.join(tmp_path, "out")
    paths = gio.to_parquet(expr, outdir, steps_per_file=2)
    assert len(paths) == 2 * len(LENGTHS)
    assert paths == sorted(paths)
    back = np.concatenate([pq.read_table(p)["data"].to_numpy() for p in paths])
    np.testing.assert_array_equal(back, whole["x"] * 2.0 + whole["n"])


def test_disabled_write_graph_run_later_equals_enabled_run(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    s = _s()
    g = gio.from_parquet(s, "events", where)
    expr = g["x"] + 0.5

    enabled = gio.to_parquet(expr, os.path.join(tmp_path, "en"), steps_per_file=1)
    plan = gio.to_parquet(expr, os.path.join(tmp_path, "dis"), steps_per_file=1, compute=False)
    assert isinstance(plan, Plan)
    later = SequentialRunner().run(plan).value
    assert [os.path.basename(p) for p in later] == [os.path.basename(p) for p in enabled]
    for a, b in zip(enabled, later, strict=True):
        np.testing.assert_array_equal(
            pq.read_table(a)["data"].to_numpy(), pq.read_table(b)["data"].to_numpy()
        )


def test_write_read_list_is_wired_from_the_projection(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    s = _s()
    g = gio.from_parquet(s, "events", where)
    expr = g["x"] * 3.0  # touches ONLY x
    plan = gio.to_parquet(expr, os.path.join(tmp_path, "o"), compute=False)
    assert isinstance(plan, Plan)
    assert set(plan.process.columns) == {"x"}  # type: ignore[attr-defined]


def test_write_rejects_multi_source_and_non_rectilinear_outputs(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    s = _s()
    a = gio.from_parquet(s, "a", where)
    b = from_record(s, "b", x=np.arange(3.0))
    with pytest.raises(TypeError, match="exactly one"):
        gio.to_parquet(a["x"] + b["x"], os.path.join(tmp_path, "nope"))

    s2 = _s()
    m = gn.from_array(s2, "m", np.ones((4, 3)))  # 2-D output is not a parquet column
    with pytest.raises(TypeError, match=r"1-D|rectilinear"):
        gio.to_parquet(np.sqrt(m), os.path.join(tmp_path, "nope2"))


def test_in_memory_record_sources_write_by_steps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = _s()
    g = from_record(s, "mem", x=np.arange(9.0), y=np.arange(9.0) * 2.0)
    paths = gio.to_parquet(g["x"] + g["y"], os.path.join(tmp_path, "mem"), steps_per_file=3)
    assert len(paths) == 3
    back = np.concatenate([pq.read_table(p)["data"].to_numpy() for p in paths])
    np.testing.assert_array_equal(back, np.arange(9.0) * 3.0)


def _ir(where: str) -> bytes:
    s = _s()
    g = gio.from_parquet(s, "events", where)
    return s.serialized_ir((g["x"] + g["n"]).reduce("sum"))


def test_parquet_programs_are_deterministic(dataset) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    assert _ir(where) == _ir(where)
