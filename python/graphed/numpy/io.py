"""Partitioned parquet I/O for the numpy backend — RECTILINEAR ONLY (M15.3, parity plan).

Specializes the backend-agnostic `graphed.parquet` base for flat columnar data: every parquet
column must be a fixed-width primitive (numeric/bool). Jagged or nested columns are REFUSED at
construction with an error naming the offending column — numpy bags cannot represent them
honestly; that data belongs to graphed-awkward. (This is the user-directed 2026-06-10 amendment
of the R16.7 deferral: parquet alone enters the numpy MVP; the rest of P3.9/P4 stays Phase 2.)

`to_parquet` follows the same R15.4/R7.8 contract as the awkward specialization: the array's
graph is compiled ONCE at the driver; each write task reads only the projected columns (the M5
field-touch projection — flat columns have no structure-only needs, so the column view is exact
here), evaluates the compiled IR, and writes one single-column parquet part.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import numpy as np

from graphed import (
    Backend,
    CompiledGraph,
    Session,
    compile_ir,
    evaluate_ir,
    refuse_chunk_partials,
)
from graphed import parquet as gpq
from graphed import write as gw
from graphed.aggregate import shipped_evaluators
from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner, WorkerResources
from graphed.errors import GraphedError
from graphed.session import _reader
from graphed.varied import Varied

from . import NumpyBackend, project
from .forms import NumpyForm


def _pa() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised in the graphed base suite
        raise ImportError(
            "parquet I/O needs pyarrow — install the optional extra: pip install "
            "'graphed[parquet]'"
        ) from exc
    return pa


# arrow primitive name -> numpy dtype, explicitly (DataType.to_pandas_dtype routes through
# pyarrow's PANDAS SHIM and fails without pandas — pandas is NOT a dependency of this package)
_ARROW_TO_NUMPY = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "halffloat": "float16",
    "float": "float32",
    "double": "float64",
}


def _rectilinear_fields(
    paths: Sequence[str], columns: Sequence[str] | None
) -> tuple[tuple[str, str], ...]:
    """(column, numpy dtype str) for every selected schema column; REFUSES non-primitives."""
    pa = _pa()
    schema = gpq.schema_of(paths)
    names = [n for n in schema.names if columns is None or n in columns]
    fields: list[tuple[str, str]] = []
    for name in names:
        typ = schema.field(name).type
        if not (pa.types.is_integer(typ) or pa.types.is_floating(typ) or pa.types.is_boolean(typ)):
            raise TypeError(
                f"column {name!r} has non-rectilinear type {typ} — numpy bags hold fixed-width "
                "primitives only; jagged/nested data belongs to graphed-awkward"
            )
        fields.append((name, np.dtype(_ARROW_TO_NUMPY[str(typ)]).str))
    return tuple(fields)


@dataclass(frozen=True)
class _DatasetLoader:
    """Lazy whole-dataset loader for the reference ``materialize`` (executors read partitions)."""

    paths: tuple[str, ...]
    columns: tuple[str, ...] | None
    steps_per_file: int = 1
    open_files: bool = False

    def source_identity(self) -> object:
        """Return what this loader reads and how it partitions it."""
        return (type(self), self.paths, self.columns, self.steps_per_file, self.open_files)

    def __call__(self) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        cols = list(self.columns) if self.columns else None
        tables = [pq.read_table(p, columns=cols) for p in self.paths]
        names = tables[0].column_names
        return {n: np.concatenate([_column_to_numpy(t, n) for t in tables]) for n in names}


@overload
def from_parquet(
    session: Session,
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = False,
) -> Any: ...
@overload
def from_parquet(
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = False,
    session: Session | None = None,
) -> Any: ...
@_reader(NumpyBackend)
def from_parquet(
    name: str,
    path: str | Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    steps_per_file: int = 1,
    open_files: bool = False,
    session: Session | None = None,
) -> Any:
    """Return a deferred RECTILINEAR record source over a parquet dataset.

    The dataset is a file / dir / glob / list; the source records into ``session``, else the
    current Session.

    The form comes from the schema alone; non-primitive columns are refused by name.
    ``steps_per_file`` and ``open_files`` are the dataset's default partitioning, as in
    ``graphed.awkward.from_parquet``, taken by ``to_parquet`` unless it is given its own.
    """
    paths = gpq.discover(path)
    if steps_per_file < 1:
        raise ValueError(f"steps_per_file must be >= 1, got {steps_per_file}")
    fields = _rectilinear_fields(paths, columns)
    form = NumpyForm(np.dtype(object), kind="record", fields=fields)
    loader = _DatasetLoader(paths, tuple(columns) if columns else None, steps_per_file, open_files)
    return gpq.deferred_source(
        name,
        paths=paths,
        form=form,
        loader=loader,
        session=session,
        steps_per_file=steps_per_file,
        open_files=open_files,
    )


def _column_to_numpy(table: Any, name: str) -> np.ndarray:
    """pandas-free column conversion.

    ChunkedArray.to_numpy routes through pyarrow's PANDAS SHIM (found in CI, where pandas is absent
    — locally a sibling repo's dev deps masked it); Array.to_numpy after combine_chunks does not.
    """
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)  # type: ignore[no-any-return]


def read_parquet_partition(
    partition: Partition, columns: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Read one partition (resolving blind ones), restricted to ``columns``."""
    import pyarrow.parquet as pq

    part = gpq.resolve_partition(partition)
    table = pq.read_table(part.uri, columns=list(columns) if columns else None)
    return {
        n: _column_to_numpy(table, n)[part.entry_start : part.entry_stop]
        for n in table.column_names
    }


# ---- deferred writing ------------------------------------------------------------------------
@dataclass(frozen=True)
class _WritePart:
    """The picklable per-partition write task: compiled IR in, one single-column part out."""

    compiled: CompiledGraph
    source_name: str
    columns: tuple[str, ...]
    destination: str
    prefix: str
    column: str
    steps_per_file: int
    bases: tuple[tuple[str, int], ...]
    memory_data: tuple[tuple[str, np.ndarray], ...] | None = None
    memory_rows: int = 0
    externals: tuple[tuple[str, Callable[..., object]], ...] = ()

    def part_paths(self, partition: Partition) -> list[str]:
        if self.memory_data is not None:
            index = _memory_step(partition, self.memory_rows, self.steps_per_file)
        else:
            index = gpq.derive_part_index(
                partition, steps_per_file=self.steps_per_file, bases=dict(self.bases)
            )
        return [gpq.part_path(self.destination, index, prefix=self.prefix)]

    def __call__(self, partition: Partition, resources: WorkerResources) -> list[str]:
        chunk: object
        if self.memory_data is not None:
            sliced = {
                k: v[partition.entry_start : partition.entry_stop] for k, v in self.memory_data
            }
            # a single unnamed entry is a FLAT source: the chunk IS the array, not a record
            chunk = sliced[""] if set(sliced) == {""} else sliced
        else:
            chunk = read_parquet_partition(partition, self.columns or None)
        (out,) = evaluate_ir(
            self.compiled,
            cast("Backend", NumpyBackend()),
            {self.source_name: chunk},
            externals=dict(self.externals),
        )
        result = np.asarray(out)
        if result.ndim != 1:
            raise TypeError(
                f"to_parquet writes 1-D rectilinear columns; the result has shape {result.shape}"
            )
        pa = _pa()
        import pyarrow.parquet as pq

        (path,) = self.part_paths(partition)
        fs, where = gw.prepare_part(path)
        pq.write_table(pa.table({self.column: result}), where, filesystem=fs)
        return [path]


def _memory_step(partition: Partition, n: int, steps: int) -> int:
    for s in range(steps):
        if ((s * n) // steps, ((s + 1) * n) // steps) == (
            partition.entry_start,
            partition.entry_stop,
        ):
            return s
    raise ValueError(f"{partition} does not match any of {steps} steps over {n} rows")


@overload
def to_parquet(
    array: Any,
    destination: str,
    *,
    steps_per_file: int = ...,
    compute: Literal[True] = ...,
    executor: Any | None = ...,
    prefix: str = ...,
    column: str = ...,
) -> list[str]: ...
@overload
def to_parquet(
    array: Any,
    destination: str,
    *,
    steps_per_file: int = ...,
    compute: Literal[False],
    executor: Any | None = ...,
    prefix: str = ...,
    column: str = ...,
) -> Plan[list[str]]: ...
@overload
def to_parquet(
    array: Any,
    destination: str,
    *,
    steps_per_file: int = ...,
    compute: bool,
    executor: Any | None = ...,
    prefix: str = ...,
    column: str = ...,
) -> list[str] | Plan[list[str]]: ...
def to_parquet(
    array: Any,
    destination: str,
    *,
    steps_per_file: int | None = None,
    compute: bool = True,
    executor: Any | None = None,
    prefix: str = "part",
    column: str = "data",
) -> list[str] | Plan[list[str]]:
    """Write the deferred 1-D array to parquet parts, one per partition (R15.4 semantics).

    With ``compute=False`` returns the task graph of write tasks; ``compute=True`` runs that SAME
    plan (sequential reference runner by default; any R7 executor pluggable). Exactly one source;
    the per-task read list is the M5 field-touch projection (exact for flat columns). A
    ``destination`` containing ``://`` is an fsspec URL: the parts are written on its filesystem.
    ``steps_per_file`` parts are written per input file; ``None`` takes ``from_parquet``'s.
    """
    if isinstance(array, Varied):
        # §6.4f: the numpy idiom hard-caps output at one 1-D column and gains no variation-aware
        # write-out — a `Varied` first positional is refused here rather than dying on §2.2's
        # reserved-name AttributeError, pointing the analyst at the backend that DOES write
        # variations.
        raise GraphedError(
            "graphed.numpy.io.to_parquet does not write variations; a numpy bag holds one 1-D "
            "column — write varied skims through the awkward backend (graphed.awkward.to_parquet)"
        )
    session: Session = array.session
    sources = gw.sources_read([array])
    if len(sources) != 1:
        raise TypeError(
            f"to_parquet needs an array recorded over exactly one source, got {len(sources)}"
        )
    ((node_id, data),) = sources.items()
    if steps_per_file is None:
        steps_per_file = data.steps_per_file if isinstance(data, _DatasetLoader) else 1
    source_name = session.source_name(node_id)
    columns = tuple(sorted(project(array, on_fail="pass").columns_for(source_name)))
    compiled = compile_ir(session, array)
    refuse_chunk_partials(compiled, as_outputs=True)

    common = {
        "compiled": compiled,
        "externals": shipped_evaluators(session, compiled),
        "source_name": source_name,
        "columns": columns,
        "destination": destination,
        "prefix": prefix,
        "column": column,
        "steps_per_file": steps_per_file,
    }
    if isinstance(data, _DatasetLoader):
        partitions = gpq.make_partitions(
            data.paths, steps_per_file=steps_per_file, open_files=data.open_files
        )
        writer = _WritePart(
            bases=tuple(gpq.file_bases(data.paths, steps_per_file).items()),  # type: ignore[arg-type]
            **common,  # type: ignore[arg-type]
        )
    else:
        resolved = data() if callable(data) else data
        if isinstance(resolved, dict):
            mem = tuple((k, np.asarray(v)) for k, v in resolved.items())
            n = len(mem[0][1])
        else:
            mem = (("", np.asarray(resolved)),)
            n = len(mem[0][1])
        partitions = tuple(
            Partition(f"memory://{source_name}", "", start, stop)
            for start, stop in gw.eager_ranges(n, steps_per_file)
        )
        writer = _WritePart(bases=(), memory_data=mem, memory_rows=n, **common)  # type: ignore[arg-type]

    plan = gpq.write_plan(partitions, writer)
    if not compute:
        return plan
    runner = executor if executor is not None else SequentialRunner()
    return list(runner.run(plan).value)
