"""Backend-agnostic deferred parquet I/O — the common base (M15.1, dask-awkward parity plan).

The pieces every parquet specialization shares, with NO array-library content (plan §A.4 —
graphed stays numpy/awkward-free; the array codecs live in `graphed_awkward.io` and
`graphed_numpy.io`):

- **Discovery** is deterministic: a directory or glob is SORTED; an explicit list keeps the
  caller's order (it is part of the dataset's identity).
- **Partitioning** uses the first-class blind `Partition` (R7.9): with ``open_files=False`` no
  file is opened at partition time — the entry range resolves at read time against the file's
  metadata row count.
- **Deferred sources** record one source per dataset whose identity carries the file list and
  whose data is a LAZY whole-dataset loader (used only by the reference ``materialize``;
  executors read per-partition through the specializations' readers instead).
- **The deferred write plan** (R15.4 semantics): compute-disabled returns a task graph of write
  tasks — each writes one output part and returns its path — and running that same plan IS the
  compute-enabled mode. The `SequentialRunner` here is the dependency-free reference; any R7
  executor (e.g. graphed-exec-local's process pool) accepts the same plan.
- **Part naming** (R15.9): a writer derives its own output-part index from its partition plus a
  per-file base table, so per-task pickled state is bounded by the number of FILES, not
  partitions.

pyarrow is an OPTIONAL dependency (`graphed[parquet]`), imported lazily and only for
file/metadata/schema handling — never for array decoding.
"""

from __future__ import annotations

import glob as _glob
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from graphed_core import Partition
from graphed_core.execution import ExecResult, Plan, Task, WorkerResources

from .array import Array
from .backend import Form, ParamValue
from .session import Session

PathLike = str | Iterable[str]


def _pq() -> Any:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415  (lazy: pyarrow is the optional extra)
    except ImportError as exc:  # pragma: no cover - exercised via the import-hook test
        raise ImportError(
            "parquet I/O needs pyarrow — install the optional extra: pip install 'graphed[parquet]'"
        ) from exc
    return pq


# ---- discovery & metadata --------------------------------------------------------------------
def discover(path: PathLike) -> tuple[str, ...]:
    """Resolve a file / directory / glob / explicit list to a deterministic tuple of paths.

    Directories and globs are sorted; an explicit list keeps the caller's order. No file is
    opened. Raises ``FileNotFoundError`` when nothing matches."""
    if isinstance(path, str):
        if os.path.isdir(path):
            found = sorted(_glob.glob(os.path.join(path, "*.parquet")))
        elif any(ch in path for ch in "*?["):
            found = sorted(_glob.glob(path))
        else:
            found = [path]
    else:
        found = [str(p) for p in path]
    if not found:
        raise FileNotFoundError(f"no parquet files match {path!r}")
    return tuple(found)


def num_rows(path: str) -> int:
    """The file's row count, from parquet METADATA only (no column data is read)."""
    return int(_pq().ParquetFile(path).metadata.num_rows)


def schema_of(paths: Sequence[str]) -> Any:
    """The dataset's arrow schema — the FIRST file is authoritative (MVP convention)."""
    return _pq().ParquetFile(paths[0]).schema_arrow


# ---- partitioning ----------------------------------------------------------------------------
def make_partitions(
    paths: Sequence[str], *, steps_per_file: int = 1, open_files: bool = True
) -> tuple[Partition, ...]:
    """``steps_per_file`` partitions per file. With ``open_files=False`` the partitions are BLIND
    (first-class, R7.9): no file is opened here; ranges resolve at read time."""
    if steps_per_file < 1:
        raise ValueError(f"steps_per_file must be >= 1, got {steps_per_file}")
    out: list[Partition] = []
    for p in paths:
        if open_files:
            n = num_rows(p)
            out.extend(
                Partition(p, "", (s * n) // steps_per_file, ((s + 1) * n) // steps_per_file)
                for s in range(steps_per_file)
            )
        else:
            out.extend(Partition.blind(p, "", s, steps_per_file) for s in range(steps_per_file))
    return tuple(out)


def resolve_partition(partition: Partition) -> Partition:
    """Resolve a blind partition against its file's metadata row count (non-blind: unchanged)."""
    if not partition.is_blind:
        return partition
    return partition.resolve(num_rows(partition.uri))


# ---- writer-side part naming (R15.9) ----------------------------------------------------------
def file_bases(paths: Sequence[str], steps_per_file: int) -> dict[str, int]:
    """uri -> first output-part index of that file (per-task state bounded by FILES)."""
    return {p: i * steps_per_file for i, p in enumerate(paths)}


def derive_part_index(partition: Partition, *, steps_per_file: int, bases: Mapping[str, int]) -> int:
    """The output-part index a writer derives from ITS OWN partition: file base + step.

    Blind partitions carry their step; an eager partition's step is reconstructed exactly by
    re-deriving the file's step ranges (entry_start alone is not invertible — n=5/steps=3 gives
    starts 0,1,3)."""
    base = bases[partition.uri]
    if partition.is_blind:
        assert partition.blind_step is not None
        return base + partition.blind_step
    n = num_rows(partition.uri)
    for s in range(steps_per_file):
        if ((s * n) // steps_per_file, ((s + 1) * n) // steps_per_file) == (
            partition.entry_start,
            partition.entry_stop,
        ):
            return base + s
    raise ValueError(f"{partition} does not match any of {steps_per_file} steps over {n} rows")


def part_path(destination: str, index: int, *, prefix: str = "part") -> str:
    return os.path.join(destination, f"{prefix}-{index:05d}.parquet")


# ---- deferred sources --------------------------------------------------------------------------
def deferred_source(
    session: Session,
    name: str,
    *,
    paths: Sequence[str],
    form: Form,
    loader: Callable[[], object],
    **params: ParamValue,
) -> Array:
    """Record one source for the whole dataset: the file list is part of the source's recorded
    identity (different files => different node), the data is the LAZY loader — nothing is read
    until the reference ``materialize`` asks (executors never call it; they read per-partition)."""
    return session.source(name, form=form, data=loader, uri=";".join(paths), format="parquet", **params)


# ---- the deferred write plan --------------------------------------------------------------------
def _concat_paths(a: list[str], b: list[str]) -> list[str]:
    return [*a, *b]


def _no_paths() -> list[str]:
    return []


def write_plan(
    partitions: Sequence[Partition],
    write_part: Callable[[Partition, WorkerResources], list[str]],
) -> Plan[list[str]]:
    """A task graph of write tasks (R15.4 compute-disabled form): each task writes one output part
    and returns its path; the combine concatenates path lists up the FIXED key-ordered tree, so
    the final path list is deterministic. ``write_part`` must be picklable (a module-level
    function or frozen dataclass) so the plan runs on a process-pool executor unchanged."""
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    return Plan(process=write_part, combine=_concat_paths, empty=_no_paths, tasks=tasks)


class _LocalResources:
    """Reference WorkerResources: opens each uri once per runner."""

    def __init__(self) -> None:
        self._handles: dict[str, object] = {}

    def open_once(self, uri: str, opener: Callable[[str], object]) -> object:
        if uri not in self._handles:
            self._handles[uri] = opener(uri)
        return self._handles[uri]


class SequentialRunner:
    """The dependency-free reference runner: executes a plan's tasks IN KEY ORDER in-process.

    Exists so the compute-enabled path of `to_parquet` works without graphed-exec-local;
    any R7 executor accepting the same plan may be passed instead."""

    def run(self, plan: Plan[list[str]]) -> ExecResult[list[str]]:
        resources = _LocalResources()
        value = plan.empty()
        n = 0
        for task in sorted(plan.tasks, key=lambda t: t.key):
            value = plan.combine(value, plan.process(task.partition, resources))
            n += 1
        return ExecResult(value=value, n_partitions=n, n_combines=max(0, n - 1))
