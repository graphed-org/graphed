"""Backend-agnostic deferred parquet I/O — the common base (M15.1, dask-awkward parity plan).

The pieces every parquet specialization shares, with NO array-library content (plan §A.4 —
graphed stays numpy/awkward-free; the array codecs live in `graphed.awkward.io` and
`graphed.numpy.io`):

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
  compute-enabled mode. Run the plan with graphed.core.execution.SequentialRunner (dependency-free reference) or any R7
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

from graphed.core import Partition

from .array import Array
from .backend import Form, ParamValue
from .session import Session
from .write import blind_part_index, file_bases, step_of, write_plan
from .write import part_path as _base_part_path

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


# ---- writer-side part naming (R15.9): the graphed.write base, parquet-flavored -------------------
def derive_part_index(partition: Partition, *, steps_per_file: int, bases: Mapping[str, int]) -> int:
    """The output-part index a writer derives from ITS OWN partition: file base + step.

    Blind partitions carry their step (no I/O); an eager partition's step is reconstructed
    exactly against the file's metadata row count (``graphed.write.step_of``)."""
    if partition.is_blind:
        return blind_part_index(partition, bases)
    n = num_rows(partition.uri)
    return bases[partition.uri] + step_of(partition.entry_start, partition.entry_stop, n, steps_per_file)


def part_path(destination: str, index: int, *, prefix: str = "part") -> str:
    return _base_part_path(destination, index, prefix=prefix, suffix=".parquet")


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


# ---- the deferred write plan: re-exported from the format-agnostic base (graphed.write) ----------
# write_plan / file_bases are ALIASES of graphed.write's — the M15 surface is the parquet
# specialization of the M20 base. The reference runner is graphed.core.execution.SequentialRunner
# (it is general execution, not a write/parquet concept).
__all__ = [
    "blind_part_index",
    "deferred_source",
    "derive_part_index",
    "discover",
    "file_bases",
    "make_partitions",
    "num_rows",
    "part_path",
    "resolve_partition",
    "schema_of",
    "step_of",
    "write_plan",
]
