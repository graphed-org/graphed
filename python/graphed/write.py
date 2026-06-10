"""The format-agnostic partitioned-write base (M20; user-directed refactor of parity P3.6).

Every deferred writer — parquet in the array backends, ROOT in the reader integration — shares
the same skeleton: a set of partitions becomes a TASK GRAPH of write tasks (each writes one
output part and reports its path), the compute-disabled graph run later IS the compute-enabled
mode (R15.4), workers derive their own part index from their partition plus an O(#files) base
table (R15.9 — no per-partition path map pickled into every task), and any R7 executor runs the
plan (with a dependency-free key-ordered sequential reference here). This module carries that
skeleton with NO format content; specializations supply only the array codec and naming suffix.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

from graphed_core import Partition
from graphed_core.execution import ExecResult, Plan, Task, WorkerResources


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

    Exists so a writer's compute-enabled path works without graphed-exec-local; any R7 executor
    accepting the same plan may be passed instead."""

    def run(self, plan: Plan[list[str]]) -> ExecResult[list[str]]:
        resources = _LocalResources()
        value = plan.empty()
        n = 0
        for task in sorted(plan.tasks, key=lambda t: t.key):
            value = plan.combine(value, plan.process(task.partition, resources))
            n += 1
        return ExecResult(value=value, n_partitions=n, n_combines=max(0, n - 1))


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
