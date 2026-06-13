"""M15.2: partitioned parquet I/O for the awkward backend (dask-awkward parity plan).

`from_parquet` grows multi-file/glob datasets, `steps_per_file`, blind partitioning, and a
column filter — with the form still built from the ARROW SCHEMA ALONE (witnessed: a dataset whose
second file is garbage bytes still constructs and types correctly; only materialization fails).
`read_parquet_partition` resolves blind partitions and reads only the requested columns.
`to_parquet` writes the array's graph per partition through the COMPILED IR (R7.8 — no
re-recording), reading only the PROJECTED columns, with the compute-disabled task graph equal to
the compute-enabled run bit-for-bit (R15.4).
"""

from __future__ import annotations

import os

import awkward as ak
import pytest
from graphed import Session

pytest.importorskip("pyarrow")

from graphed import parquet as gpq
from graphed_core.execution import Plan, SequentialRunner

import graphed_awkward.io as gio
from graphed_awkward import AwkwardBackend, from_awkward, from_parquet, gak, project

LENGTHS = [6, 4, 5]


def _events(n: int, offset: int = 0) -> ak.Array:
    return ak.Array(
        {
            "x": [float(offset + i) for i in range(n)],
            "Jet": [[{"pt": float(offset + i + j), "eta": 0.1 * j} for j in range(i % 3)] for i in range(n)],
        }
    )


@pytest.fixture
def dataset(tmp_path) -> tuple[str, ak.Array]:  # type: ignore[no-untyped-def]
    chunks = []
    offset = 0
    for i, n in enumerate(LENGTHS):
        chunk = _events(n, offset)
        ak.to_parquet(chunk, os.path.join(tmp_path, f"events-{i}.parquet"))
        chunks.append(chunk)
        offset += n
    return str(tmp_path), ak.concatenate(chunks)


def _s() -> Session:
    return Session(AwkwardBackend())


# ---- deferred reading ------------------------------------------------------------------------
def test_multifile_from_parquet_materializes_the_concatenation(dataset) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    g = from_parquet(s, "events", where)
    assert s.form(g).is_typetracer
    out = ak.Array(s.materialize(g))
    assert ak.array_equal(out, whole)


def test_form_comes_from_the_schema_alone(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # the witness: the SECOND file is garbage bytes — construction (schema from the first file,
    # blind partitioning) must succeed and type correctly; only materialization may fail
    where, _ = dataset
    paths = sorted(gpq.discover(where))
    garbage = os.path.join(tmp_path, "zzz-garbage.parquet")
    with open(garbage, "wb") as f:
        f.write(b"this is not parquet")
    s = _s()
    g = from_parquet(s, "events", [paths[0], garbage], open_files=False)
    assert "var *" in s.form(g.Jet.pt).describe()
    with pytest.raises(Exception):  # noqa: B017  (any decode error: the data really is garbage)
        s.materialize(g)


def test_column_filter_restricts_the_loaded_fields(dataset) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    g = from_parquet(s, "events", where, columns=["x"])
    out = ak.Array(s.materialize(g))
    assert out.fields == ["x"]
    assert ak.array_equal(out.x, whole.x)


def test_single_file_m3_shape_still_holds(dataset) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    path = sorted(gpq.discover(where))[0]
    s = _s()
    g = from_parquet(s, "events", path)
    assert s.form(g.Jet.pt).is_typetracer
    assert len(ak.Array(s.materialize(g))) == LENGTHS[0]


# ---- partition reads -------------------------------------------------------------------------
def test_partition_reads_tile_the_dataset(dataset) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    paths = gpq.discover(where)
    for open_files in (True, False):
        parts = gpq.make_partitions(paths, steps_per_file=2, open_files=open_files)
        chunks = [gio.read_parquet_partition(p) for p in parts]
        assert ak.array_equal(ak.concatenate(chunks), whole)


def test_partition_reads_project_columns(dataset) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    (first, *_rest) = gpq.discover(where)
    part = gpq.make_partitions([first], steps_per_file=1)[0]
    chunk = gio.read_parquet_partition(part, columns=["x"])
    assert chunk.fields == ["x"]


# ---- deferred writing ------------------------------------------------------------------------
def test_to_parquet_roundtrips_through_the_compiled_ir(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    g = from_parquet(s, "events", where)
    expr = gak.num(g.Jet, axis=1) + g.x
    outdir = os.path.join(tmp_path, "out")
    paths = gio.to_parquet(expr, outdir, steps_per_file=2)
    assert len(paths) == 2 * len(LENGTHS)
    assert paths == sorted(paths)  # deterministic part naming, key-ordered
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    ref = ak.num(whole.Jet, axis=1) + whole.x
    assert ak.array_equal(back["data"], ref)


def test_disabled_write_graph_run_later_equals_enabled_run(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, _whole = dataset
    s = _s()
    g = from_parquet(s, "events", where)
    expr = g.x * 2.0

    enabled_dir = os.path.join(tmp_path, "enabled")
    enabled = gio.to_parquet(expr, enabled_dir, steps_per_file=1)

    disabled_dir = os.path.join(tmp_path, "disabled")
    plan = gio.to_parquet(expr, disabled_dir, steps_per_file=1, compute=False)
    assert isinstance(plan, Plan)  # a task graph of write tasks, not outputs (R15.4)
    assert not os.path.exists(disabled_dir) or not os.listdir(disabled_dir)  # nothing written yet
    later = SequentialRunner().run(plan).value

    assert [os.path.basename(p) for p in later] == [os.path.basename(p) for p in enabled]
    for a, b in zip(enabled, later, strict=True):
        assert ak.array_equal(ak.from_parquet(a), ak.from_parquet(b))  # bit-for-bit consistent


def test_write_reads_only_the_projected_columns(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, _ = dataset
    s = _s()
    g = from_parquet(s, "events", where)
    expr = g.x + 1.0  # touches ONLY x
    proj = project(expr)
    plan = gio.to_parquet(expr, os.path.join(tmp_path, "o"), compute=False)
    assert isinstance(plan, Plan)
    # the writer's read list is WIRED FROM THE PROJECTION (R15.3): they cannot drift apart
    assert set(plan.process.columns) == set(proj.columns_for("events")) == {"x"}  # type: ignore[attr-defined]


def test_write_rejects_multi_source_arrays(dataset, tmp_path) -> None:  # type: ignore[no-untyped-def]
    where, whole = dataset
    s = _s()
    a = from_parquet(s, "a", where)
    b = from_awkward(s, "b", whole)
    with pytest.raises(TypeError, match="exactly one"):
        gio.to_parquet(a.x + b.x, os.path.join(tmp_path, "nope"))


def test_in_memory_sources_write_by_steps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    whole = _events(9)
    s = _s()
    g = from_awkward(s, "mem", whole)
    paths = gio.to_parquet(g.x * 3.0, os.path.join(tmp_path, "mem"), steps_per_file=3)
    assert len(paths) == 3
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert ak.array_equal(back["data"], whole.x * 3.0)
