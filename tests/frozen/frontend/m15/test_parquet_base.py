"""M15: the backend-agnostic parquet base (dask-awkward parity plan, M15.1).

The common machinery every parquet specialization builds on: deterministic discovery,
metadata-only row counts, blind/eager partitioning on the FIRST-CLASS blind Partition (R7.9 — a
blind partitioning opens NO file, witnessed with nonexistent paths), lazy deferred sources, and
the deferred write plan whose compute-disabled graph, when run, produces the compute-enabled
outputs (R15.4 consistency).
"""

from __future__ import annotations

import builtins
import os
from itertools import pairwise

import pytest
from m15_toy import CountingLoader, ToyForm, session

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402, I001

from graphed import parquet as gpq  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402  (M32: the reference runner moved here)
from graphed.core import Partition  # noqa: E402
from graphed.core.execution import Task, WorkerResources  # noqa: E402

ROWS = [5, 3, 7]  # rows per fixture file


@pytest.fixture
def files(tmp_path) -> list[str]:  # type: ignore[no-untyped-def]
    out = []
    for i, n in enumerate(ROWS):
        p = os.path.join(tmp_path, f"data-{i}.parquet")
        pq.write_table(pa.table({"x": list(range(n)), "y": [float(v) * 2 for v in range(n)]}), p)
        out.append(p)
    return out


# ---- discovery -----------------------------------------------------------------------------------
def test_discovery_forms(files: list[str], tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert gpq.discover(files[0]) == (files[0],)
    assert gpq.discover(str(tmp_path)) == tuple(sorted(files))  # directory: sorted
    assert gpq.discover(os.path.join(tmp_path, "data-*.parquet")) == tuple(sorted(files))  # glob: sorted
    explicit = [files[2], files[0]]
    assert gpq.discover(explicit) == (files[2], files[0])  # explicit list: caller's order kept
    with pytest.raises(FileNotFoundError):
        gpq.discover(os.path.join(tmp_path, "nope-*.parquet"))


def test_num_rows_and_schema_are_metadata_reads(files: list[str]) -> None:
    assert [gpq.num_rows(p) for p in files] == ROWS
    schema = gpq.schema_of(gpq.discover(files))
    assert schema.names == ["x", "y"]


# ---- partitioning --------------------------------------------------------------------------------
def test_blind_partitioning_opens_no_file() -> None:
    # the witness: these paths DO NOT EXIST, yet blind partitioning must succeed (R15.3/R7.9)
    ghosts = ["/definitely/not/here-0.parquet", "/definitely/not/here-1.parquet"]
    parts = gpq.make_partitions(ghosts, steps_per_file=3, open_files=False)
    assert len(parts) == 6
    assert all(p.is_blind for p in parts)
    assert [(p.blind_step, p.blind_n_steps) for p in parts[:3]] == [(0, 3), (1, 3), (2, 3)]


def test_eager_partitions_cover_each_file_contiguously(files: list[str]) -> None:
    parts = gpq.make_partitions(files, steps_per_file=2, open_files=True)
    assert len(parts) == 6
    for path, n in zip(files, ROWS, strict=True):
        ranges = sorted((p.entry_start, p.entry_stop) for p in parts if p.uri == path)
        assert ranges[0][0] == 0 and ranges[-1][1] == n
        assert all(a[1] == b[0] for a, b in pairwise(ranges))


def test_blind_resolution_equals_eager_partitioning(files: list[str]) -> None:
    eager = gpq.make_partitions(files, steps_per_file=3, open_files=True)
    blind = gpq.make_partitions(files, steps_per_file=3, open_files=False)
    resolved = [gpq.resolve_partition(p) for p in blind]
    assert sorted((p.uri, p.entry_start, p.entry_stop) for p in resolved) == sorted(
        (p.uri, p.entry_start, p.entry_stop) for p in eager
    )
    nonblind = eager[0]
    assert gpq.resolve_partition(nonblind) is nonblind  # already concrete: untouched


def test_part_index_is_stable_for_blind_and_eager(files: list[str]) -> None:
    # the writer derives ITS OWN output-part index from its partition (R15.9: no global
    # partition-to-path map pickled into every task); the awkward case n=5, steps=3 included
    steps = 3
    paths = gpq.discover(files)
    bases = gpq.file_bases(paths, steps)
    assert [bases[p] for p in paths] == [0, 3, 6]
    eager = gpq.make_partitions(paths, steps_per_file=steps, open_files=True)
    blind = gpq.make_partitions(paths, steps_per_file=steps, open_files=False)
    eager_idx = sorted(gpq.derive_part_index(p, steps_per_file=steps, bases=bases) for p in eager)
    blind_idx = sorted(gpq.derive_part_index(p, steps_per_file=steps, bases=bases) for p in blind)
    assert eager_idx == blind_idx == list(range(9))
    assert gpq.part_path("/out", 7) == os.path.join("/out", "part-00007.parquet")


# ---- deferred sources ----------------------------------------------------------------------------
def test_deferred_source_is_lazy_and_identity_carries_the_dataset(files: list[str]) -> None:
    s = session()
    loader = CountingLoader(value="DATA")
    paths = gpq.discover(files)
    arr = gpq.deferred_source(s, "events", paths=paths, form=ToyForm("source"), loader=loader)
    assert loader.calls == []  # NOTHING read at record time
    assert s.materialize(arr) == "DATA"
    assert loader.calls == [1]  # the lazy loader ran exactly once

    s2 = session()
    a = gpq.deferred_source(s2, "events", paths=paths, form=ToyForm("source"), loader=loader)
    b = gpq.deferred_source(s2, "events", paths=paths, form=ToyForm("source"), loader=loader)
    assert a.node_id == b.node_id  # same dataset interns to one source
    other = gpq.deferred_source(s2, "events", paths=paths[:1], form=ToyForm("source"), loader=loader)
    assert other.node_id != a.node_id  # a DIFFERENT file list is a different source identity


# ---- the deferred write plan ---------------------------------------------------------------------
class _SpyResources:
    def open_once(self, uri: str, opener):  # type: ignore[no-untyped-def]
        return opener(uri)


def _toy_writer(partition: Partition, resources: WorkerResources) -> list[str]:
    # module-level (picklable) writer: one text part per partition
    out = partition.uri + f".part{partition.entry_start}-{partition.entry_stop}.txt"
    with open(out, "w") as f:
        f.write(f"{partition.uri}:{partition.entry_start}:{partition.entry_stop}\n")
    return [out]


def test_write_plan_disabled_graph_run_later_matches_enabled_run(files: list[str]) -> None:
    parts = gpq.make_partitions(files, steps_per_file=1, open_files=True)
    plan = gpq.write_plan(parts, _toy_writer)
    # compute-disabled: a task graph of write tasks, NOT outputs (R15.4)
    assert [t.partition for t in plan.tasks] == list(parts)
    assert all(isinstance(t, Task) for t in plan.tasks)
    assert plan.empty() == []

    # running the disabled graph produces the enabled mode's outputs
    result = SequentialRunner().run(plan)
    assert result.n_partitions == len(parts)
    assert result.value == [p.uri + f".part{p.entry_start}-{p.entry_stop}.txt" for p in parts]
    for path in result.value:
        assert os.path.exists(path)


def test_sequential_runner_is_key_ordered_and_combine_is_plan_combine(files: list[str]) -> None:
    parts = gpq.make_partitions(files, steps_per_file=1, open_files=True)
    plan = gpq.write_plan(tuple(reversed(parts)), _toy_writer)
    result = SequentialRunner().run(plan)
    # keys fix the order regardless of the sequence handed in: output order is key order
    assert result.value == [
        t.partition.uri + f".part{t.partition.entry_start}-{t.partition.entry_stop}.txt"
        for t in sorted(plan.tasks, key=lambda t: t.key)
    ]


def test_missing_pyarrow_message_names_the_extra(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real_import = builtins.__import__

    def no_pyarrow(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("pyarrow"):
            raise ImportError("pyarrow not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    with pytest.raises(ImportError, match="graphed\\[parquet\\]"):
        gpq.num_rows("/nowhere.parquet")
