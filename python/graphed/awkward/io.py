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
from graphed import write as gw
from graphed.write import PartitionedSource
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
    """Lazy whole-dataset loader for the reference ``materialize`` — AND a
    ``graphed.write.PartitionedSource``, so the generic writer reads it partition by partition
    (the whole-dataset path is only ever the reference ``materialize``)."""

    paths: tuple[str, ...]
    columns: tuple[str, ...] | None

    def __call__(self) -> ak.Array:
        return ak.from_parquet(list(self.paths), columns=list(self.columns) if self.columns else None)

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return gpq.make_partitions(self.paths, steps_per_file=steps_per_file, open_files=False)

    def read_partition(self, partition: Partition, columns: Sequence[str] | None, resources: Any) -> ak.Array:
        return read_parquet_partition(partition, columns if columns else (self.columns or None))


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
    bases: tuple[tuple[Any, int], ...]
    column: str = "data"
    reader: Any = None  # a graphed.write.PartitionedSource (picklable)
    behavior: Any = None  # a behavior dict or an importable "module:attr" reference
    memory_data: ak.Array | None = None  # in-memory source payload (bounded by the dataset)
    memory_rows: int = 0

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        if self.reader is not None:
            chunk = self.reader.read_partition(partition, self.columns or None, resources)
            index = gw.blind_part_index(partition, dict(self.bases))
        elif self.memory_data is not None:
            chunk = self.memory_data[partition.entry_start : partition.entry_stop]
            index = _memory_step(partition, self.memory_rows, self.steps_per_file)
        else:  # pragma: no cover - every source is a protocol reader or in-memory
            raise TypeError("write task has neither a partition reader nor in-memory data")
        backend = AwkwardBackend(behavior=_resolve_behavior(self.behavior))
        (out,) = evaluate_ir(self.compiled, cast("Backend", backend), {self.source_name: chunk})
        result = ak.Array(out)
        payload = result if result.fields else ak.Array({self.column: result})
        os.makedirs(self.destination, exist_ok=True)
        path = gpq.part_path(self.destination, index, prefix=self.prefix)
        ak.to_parquet(payload, path)
        return [path]


def _syntactic_fields(array: Any, source_node_id: int) -> set[str] | None:
    """The top-level source fields the recorded graph ACCESSES (a session walk). ``None`` means
    the whole source is consumed (a bare source, or a non-field op applied to it directly)."""
    found: set[str] = set()
    whole = [False]
    sentinel = object()

    def on_source(nid: int) -> object:
        return (sentinel, nid)

    def on_op(_nid: int, name: str, ins: list[object], params: Any) -> object:
        touches_source = any(
            isinstance(x, tuple) and len(x) == 2 and x[0] is sentinel and x[1] == source_node_id for x in ins
        )
        if touches_source:
            if name == "field":
                found.add(str(params["field"]))
            elif name == "fields":
                found.update(f for f in str(params["fields"]).split(",") if f)
            else:
                whole[0] = True
        return None

    array.session.walk(array, source=on_source, op=on_op, external=lambda *_a: None)
    return None if (whole[0] or not found) else found


def _evaluation_columns(
    array: Any, source_node_id: int, source_name: str, source_form: AwkwardForm
) -> tuple[str, ...]:
    """The per-task read list: the graph's SYNTACTIC source-field accesses, refined per field by
    the buffer projection.

    Evaluation replays EVERY recorded node — including field accesses whose buffers the output
    never touches (a zip's untouched legs) — so the syntactic set decides WHICH fields must
    exist; the buffer view then decides which LEAVES to read for each: DATA needs read their
    leaves, an offsets-only field reads its CHEAPEST CARRIER leaf (parquet has no standalone
    counter column — R15.8's translation). An empty tuple means "everything" (the whole source
    is consumed)."""
    accessed = _syntactic_fields(array, source_node_id)
    if accessed is None:
        return ()
    leaves = source_form.tt.layout.form.columns()
    needs = project_buffers(array).buffers_for(source_name)
    out: set[str] = set()
    for f in sorted(accessed):
        data_paths = [
            p for p, need in needs.items() if need.value == "data" and (p == f or p.startswith(f + "."))
        ]
        if data_paths:
            for p in data_paths:
                out.update(c for c in leaves if c == p or c.startswith(p + "."))
        else:
            under = sorted(c for c in leaves if c == f or c.startswith(f + "."))
            out.add(under[0] if under else f)
    return tuple(sorted(out))


def _resolve_behavior(behavior: Any) -> Any:
    """A behavior dict, or an importable "module:attr" reference (behavior dicts often contain
    lambdas, which do not pickle to process workers)."""
    if isinstance(behavior, str):
        import importlib  # noqa: PLC0415

        mod_name, _, attr = behavior.partition(":")
        return getattr(importlib.import_module(mod_name), attr)
    return behavior


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
    column: str = "data",
    behavior: Any = None,
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
    columns = _evaluation_columns(array, node_id, source_name, source_form)
    compiled = compile_ir(session, array)

    if isinstance(data, PartitionedSource):
        # the generic path: ANY source describing its own partitioning (parquet datasets, the
        # ROOT reader integration, ...) is written partition by partition — its whole-dataset
        # loader is never invoked
        partitions = data.partitions(steps_per_file)
        keys = list(dict.fromkeys((p.uri, p.tree) if p.tree else p.uri for p in partitions))
        writer = _WritePart(
            compiled=compiled,
            source_name=source_name,
            columns=columns,
            destination=destination,
            prefix=prefix,
            steps_per_file=steps_per_file,
            bases=tuple(gw.file_bases(keys, steps_per_file).items()),
            column=column,
            reader=data,
            behavior=behavior,
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
            column=column,
            behavior=behavior,
            memory_data=whole,
            memory_rows=n,
        )

    plan = gw.write_plan(partitions, writer)
    if not compute:
        return plan
    runner = executor if executor is not None else gpq.SequentialRunner()
    return list(runner.run(plan).value)
