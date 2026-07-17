"""Retry / recovery policies for failed partitions (plan M8).

When ``process(partition, resources)`` raises, the runner asks a policy to recover. A policy returns
the partial result if it can recover, or re-raises (the runner then dead-letters the partition).
Policies (plan M8 targets): ``retry_n | retry_smaller_chunk | retry_elsewhere | quarantine``.

``RetrySmallerChunk`` is the one with teeth: it splits the partition into sub-ranges and processes
each (recursively, down to ``min_size``), combining the sub-results — so a chunk that OOMed whole
**succeeds split** (the M8 acceptance: "``retry_smaller_chunk`` succeeds where the original OOMed").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any, Protocol

from graphed.core import Partition

Process = Callable[[Partition, Any], Any]
Combine = Callable[[Any, Any], Any]


class RetryPolicy(Protocol):
    """Attempt to recover a failed partition; return the partial, or raise to dead-letter it."""

    def recover(
        self,
        partition: Partition,
        error: BaseException,
        *,
        process: Process,
        combine: Combine,
        resources: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class Quarantine:
    """Do not retry — send the partition straight to the dead-letter set."""

    def recover(
        self,
        partition: Partition,
        error: BaseException,
        *,
        process: Process,
        combine: Combine,
        resources: Any,
    ) -> Any:
        raise error


@dataclass(frozen=True)
class RetryN:
    """Retry the same partition up to ``n`` times (transient-failure policy)."""

    n: int = 1

    def recover(
        self,
        partition: Partition,
        error: BaseException,
        *,
        process: Process,
        combine: Combine,
        resources: Any,
    ) -> Any:
        last = error
        for _ in range(self.n):
            try:
                return process(partition, resources)
            except Exception as exc:
                last = exc
        raise last


@dataclass(frozen=True)
class RetryElsewhere:
    """Re-run on a fresh worker context (single-machine MVP: a new resources object per attempt).

    Models moving a task off a flaky worker; ``new_resources`` supplies the replacement context.
    """

    attempts: int = 1
    new_resources: Callable[[], Any] | None = None

    def recover(
        self,
        partition: Partition,
        error: BaseException,
        *,
        process: Process,
        combine: Combine,
        resources: Any,
    ) -> Any:
        last = error
        for _ in range(self.attempts):
            res = self.new_resources() if self.new_resources is not None else resources
            try:
                return process(partition, res)
            except Exception as exc:
                last = exc
        raise last


@dataclass(frozen=True)
class RetrySmallerChunk:
    """Split the partition into ``splits`` sub-ranges and process each (recursively, to ``min_size``),
    combining the sub-results. Recovers a chunk too big to process whole (a simulated OOM)."""

    splits: int = 2
    min_size: int = 1

    def recover(
        self,
        partition: Partition,
        error: BaseException,
        *,
        process: Process,
        combine: Combine,
        resources: Any,
    ) -> Any:
        return self._process_split(partition, error, process=process, combine=combine, resources=resources)

    def _process_split(
        self,
        partition: Partition,
        error: BaseException,
        *,
        process: Process,
        combine: Combine,
        resources: Any,
    ) -> Any:
        subs = _split(partition, self.splits)
        if len(subs) <= 1 or partition.n_entries <= self.min_size:
            # cannot shrink further: the failure is not a size problem
            raise error
        partials = []
        for sub in subs:
            try:
                partials.append(process(sub, resources))
            except Exception as exc:
                partials.append(
                    self._process_split(sub, exc, process=process, combine=combine, resources=resources)
                )
        return reduce(combine, partials)


def _split(p: Partition, n: int) -> list[Partition]:
    """Deterministically split a partition's entry range into ``<= n`` contiguous sub-partitions."""
    total = p.n_entries
    if total <= 1:
        return [p]
    n = max(2, min(n, total))
    size = total // n
    bounds = [p.entry_start + i * size for i in range(n)] + [p.entry_stop]
    out = []
    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        if hi > lo:
            out.append(Partition(p.uri, p.tree, lo, hi))
    return out
