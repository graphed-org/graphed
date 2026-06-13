"""The format-agnostic partitioned-write base (M20; user-directed refactor of parity P3.6).

Every deferred writer — parquet in the array backends, ROOT in the reader integration — shares
the same skeleton: a set of partitions becomes a TASK GRAPH of write tasks (each writes one
output part and reports its path), the compute-disabled graph run later IS the compute-enabled
mode (R15.4), workers derive their own part index from their partition plus an O(#files) base
table (R15.9 — no per-partition path map pickled into every task), and any R7 executor runs the
plan (the dependency-free key-ordered reference is ``graphed_core.execution.SequentialRunner``).
This module carries that skeleton with NO format content; specializations supply only the array
codec and naming suffix.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from graphed_core import Partition
from graphed_core.execution import Plan, Task, WorkerResources


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
    function, ``functools.partial`` of one, or a frozen dataclass) so the plan runs on a
    process-pool executor unchanged."""
    tasks = tuple(Task(i, p) for i, p in enumerate(partitions))
    return Plan(process=write_part, combine=_concat_paths, empty=_no_paths, tasks=tasks)


# ---- writer-side part naming and indexing (R15.9) ------------------------------------------------
def file_bases(keys: Sequence[Hashable], steps_per_file: int) -> dict[Hashable, int]:
    """key -> first output-part index of that file. Keys are GENERIC: a plain uri for flat-file
    formats, a (uri, tree) pair for container formats holding several objects per file."""
    return {k: i * steps_per_file for i, k in enumerate(keys)}


def blind_part_index(partition: Partition, bases: Mapping[Any, int]) -> int:
    """The output-part index a worker derives from ITS OWN blind partition: file base + blind
    step — no I/O. The base table may be keyed by ``(uri, tree)`` or plain ``uri``."""
    if partition.blind_step is None:
        raise ValueError(f"{partition} is not blind; derive its step with step_of() instead")
    pair = (partition.uri, partition.tree)
    if pair in bases:
        return bases[pair] + partition.blind_step
    return bases[partition.uri] + partition.blind_step


def step_of(entry_start: int, entry_stop: int, n_entries: int, steps_per_file: int) -> int:
    """Reconstruct which step an EAGER partition is, exactly — ``entry_start`` alone is not
    invertible (n=5, steps=3 gives starts 0, 1, 3)."""
    for s in range(steps_per_file):
        if ((s * n_entries) // steps_per_file, ((s + 1) * n_entries) // steps_per_file) == (
            entry_start,
            entry_stop,
        ):
            return s
    raise ValueError(
        f"({entry_start}, {entry_stop}) matches none of {steps_per_file} steps over {n_entries} rows"
    )


def part_path(destination: str, index: int, *, prefix: str = "part", suffix: str) -> str:
    """Deterministic part naming; the SUFFIX is the specialization's (".parquet", ".root", ...)."""
    return os.path.join(destination, f"{prefix}-{index:05d}{suffix}")


# ---- the partitioned-source protocol (read side of the base) -------------------------------------
@runtime_checkable
class PartitionedSource(Protocol):
    """A source DATA object that can be read partition by partition — the read-side counterpart of
    the write plan. Writers (and any partition-wise consumer) dispatch on this protocol instead of
    materializing the whole dataset through the source's lazy loader: ``partitions`` describes the
    dataset's partitioning (BLIND preferred — R7.9: no file opened at planning time) and
    ``read_partition`` reads exactly one partition, restricted to ``columns`` (``None`` = the
    source's own selection), with ``resources.open_once`` available for the file-locality
    directive. Implemented by the parquet dataset loader and the ROOT reader integration's source."""

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]: ...

    def read_partition(
        self, partition: Partition, columns: Sequence[str] | None, resources: WorkerResources
    ) -> object: ...
