"""Partitioned parquet I/O for the awkward backend (M15.2, dask-awkward parity plan).

Specializes the backend-agnostic `graphed.parquet` base: the awkward pieces are exactly two —
the FORM comes from the arrow schema alone (`ak.from_arrow_schema`; no event data is read at
construction) and the per-partition codec is `ak.from_parquet`/`ak.to_parquet`.

`to_parquet` follows the R15.4/R15.5 contract proven by the uproot integration: compute-disabled
returns a task graph of write tasks; each task evaluates the array's graph through the COMPILED
IR (R7.8 — compiled once at the driver, never re-recorded per partition), reads only the
PROJECTED columns (R15.3 — the read list is wired from the projection), derives its own output
part index from its partition (R15.9), and writes one parquet part.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import awkward as ak
from graphed import Backend, CompiledGraph, Session, compile_ir, evaluate_ir
from graphed import parquet as gpq
from graphed_core import Partition
from graphed_core.execution import Plan, WorkerResources

from .backend import AwkwardBackend, AwkwardForm
from .projection import project_buffers


def _schema_form(paths: Sequence[str], columns: Sequence[str] | None) -> AwkwardForm:
    """The dataset's form from the arrow SCHEMA alone (first file authoritative; nothing decoded)."""
    form = ak.from_arrow_schema(gpq.schema_of(paths))
    if columns:
        form = form.select_columns(list(columns))
    tt = ak.Array(form.length_zero_array(highlevel=False).to_typetracer(forget_length=True))
    return AwkwardForm(tt)


@dataclass(frozen=True)
class _DatasetLoader:
    """Lazy whole-dataset loader for the reference ``materialize`` (executors read partitions)."""

    paths: tuple[str, ...]
    columns: tuple[str, ...] | None

    def __call__(self) -> ak.Array:
        return ak.from_parquet(list(self.paths), columns=list(self.columns) if self.columns else None)


def from_parquet(
    session: Session,
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = True,
) -> Any:
    """A deferred awkward array over a parquet dataset (file / directory / glob / list).

    The form is built from METADATA alone; no event data is read here. ``steps_per_file`` and
    ``open_files`` shape the dataset's default partitioning (``partitions_of``); with
    ``open_files=False`` no file is opened at all (blind partitions, R7.9)."""
    paths = gpq.discover(path)
    form = _schema_form(paths, columns)
    if steps_per_file < 1:
        raise ValueError(f"steps_per_file must be >= 1, got {steps_per_file}")
    if not open_files:
        gpq.make_partitions(paths, steps_per_file=steps_per_file, open_files=False)  # validated blind
    loader = _DatasetLoader(paths, tuple(columns) if columns else None)
    return gpq.deferred_source(session, name, paths=paths, form=form, loader=loader)


def read_parquet_partition(partition: Partition, columns: Sequence[str] | None = None) -> ak.Array:
    """Read one partition (resolving blind ones at read time), restricted to ``columns``."""
    part = gpq.resolve_partition(partition)
    arr = ak.from_parquet(part.uri, columns=list(columns) if columns else None)
    return arr[part.entry_start : part.entry_stop]


# ---- deferred writing ------------------------------------------------------------------------
@dataclass(frozen=True)
class _WritePart:
    """The picklable per-partition write task: compiled IR in, one parquet part out."""

    compiled: CompiledGraph
    source_name: str
    columns: tuple[str, ...]
    destination: str
    prefix: str
    steps_per_file: int
    bases: tuple[tuple[str, int], ...]
    memory_data: ak.Array | None = None  # in-memory source payload (bounded by the dataset)
    memory_rows: int = 0

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        if self.memory_data is not None:
            chunk = self.memory_data[partition.entry_start : partition.entry_stop]
            index = _memory_step(partition, self.memory_rows, self.steps_per_file)
        else:
            chunk = read_parquet_partition(partition, self.columns or None)
            index = gpq.derive_part_index(
                partition, steps_per_file=self.steps_per_file, bases=dict(self.bases)
            )
        (out,) = evaluate_ir(self.compiled, cast("Backend", AwkwardBackend()), {self.source_name: chunk})
        result = ak.Array(out)
        payload = result if result.fields else ak.Array({"data": result})
        os.makedirs(self.destination, exist_ok=True)
        path = gpq.part_path(self.destination, index, prefix=self.prefix)
        ak.to_parquet(payload, path)
        return [path]


def _read_columns(array: Any, source_name: str, source_form: AwkwardForm) -> tuple[str, ...]:
    """The per-task parquet read list, wired from the BUFFER projection (R15.3/R15.8).

    The column view alone under-specifies structure-only needs (`gak.num(jets)` reads no leaf
    DATA, yet needs Jet's offsets — the A.3 finding M10 fixed for ROOT). Translation to what
    parquet can read: a DATA need reads that column; an OFFSETS need reads its CHEAPEST CARRIER —
    the first schema leaf at-or-under the path (any single leaf of a list materializes the list
    lengths; parquet has no standalone counter column)."""
    needs = project_buffers(array).buffers_for(source_name)
    leaves = source_form.tt.layout.form.columns()
    out: set[str] = set()
    for path, need in needs.items():
        if need.value == "data":
            out.add(path)
        else:
            under = sorted(c for c in leaves if c == path or c.startswith(path + "."))
            out.add(under[0] if under else path)
    return tuple(sorted(out))


def _memory_step(partition: Partition, n: int, steps: int) -> int:
    for s in range(steps):
        if ((s * n) // steps, ((s + 1) * n) // steps) == (partition.entry_start, partition.entry_stop):
            return s
    raise ValueError(f"{partition} does not match any of {steps} steps over {n} rows")


def to_parquet(
    array: Any,
    destination: str,
    *,
    steps_per_file: int = 1,
    compute: bool = True,
    executor: Any | None = None,
    prefix: str = "part",
) -> list[str] | Plan[list[str]]:
    """Write the deferred array to parquet parts, one per partition (R15.4 semantics).

    With ``compute=False`` returns the task graph of write tasks; with ``compute=True`` runs that
    SAME plan (``SequentialRunner`` by default; pass any R7 executor). The array must be recorded
    over exactly one source; the per-task read list comes from the recorded graph's projection."""
    session: Session = array.session
    sources = session.sources()
    if len(sources) != 1:
        raise TypeError(f"to_parquet needs an array recorded over exactly one source, got {len(sources)}")
    ((node_id, data),) = sources.items()
    source_name = session.source_name(node_id)
    source_form = session.form_of(node_id)
    assert isinstance(source_form, AwkwardForm)  # this backend recorded the source
    columns = _read_columns(array, source_name, source_form)
    compiled = compile_ir(session, array)

    if isinstance(data, _DatasetLoader):
        paths = data.paths
        partitions = gpq.make_partitions(paths, steps_per_file=steps_per_file, open_files=False)
        writer = _WritePart(
            compiled=compiled,
            source_name=source_name,
            columns=columns,
            destination=destination,
            prefix=prefix,
            steps_per_file=steps_per_file,
            bases=tuple(gpq.file_bases(paths, steps_per_file).items()),
        )
    else:
        whole = ak.Array(data() if callable(data) else data)
        n = len(whole)
        partitions = tuple(
            Partition(
                f"memory://{source_name}", "", (s * n) // steps_per_file, ((s + 1) * n) // steps_per_file
            )
            for s in range(steps_per_file)
        )
        writer = _WritePart(
            compiled=compiled,
            source_name=source_name,
            columns=columns,
            destination=destination,
            prefix=prefix,
            steps_per_file=steps_per_file,
            bases=(),
            memory_data=whole,
            memory_rows=n,
        )

    plan = gpq.write_plan(partitions, writer)
    if not compute:
        return plan
    runner = executor if executor is not None else gpq.SequentialRunner()
    return list(runner.run(plan).value)
