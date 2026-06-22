"""The resumable runner (plan M8): execute a ``DurablePlan`` against a checkpoint ``Store``.

Correctness model (why resume is safe):

- A task's identity is the plan's **content-addressed** ``task_id`` (SHA-256 over the IR + process
  spec + partition). Before running a partition the runner checks the Store; if its ``task_id`` is
  already journaled with a present blob, the expensive ``process`` is **skipped** and the stored
  partial is reused. So a resumed run does **measurably less work** (``skipped`` is logged).
- The final reduction recombines the **per-task partials** (in deterministic task order) — never a
  persisted running accumulator. Each partition therefore contributes **exactly once** regardless of
  where a crash happened: **no double-count, no lost partition**, and the result is bit-for-bit equal
  to an uninterrupted run.
- A failed partition is recovered by the ``retry`` policy or harvested into the Store's dead-letter
  set with a reproducible descriptor; an **error budget** is a stopping condition.

This is single-machine and local-filesystem only (the M8 guardrail).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

from graphed_core import DurablePlan, Partition

from .codec import Codec, PickleCodec
from .errors import dead_letter_descriptor
from .retry import RetryPolicy


@dataclass
class ResumeReport:
    """What a resumable run did — the audit trail for "measurably less work" and error harvesting."""

    executed: int = 0  # partitions whose process actually ran this invocation
    skipped: int = 0  # partitions reused from the checkpoint store
    dead: int = 0  # partitions harvested into the dead-letter set
    stopped: str | None = None  # a StopReason value if a stopping condition fired (e.g. error budget)
    dead_letters: list[dict[str, Any]] = field(default_factory=list)

    @property
    def did_less_work(self) -> bool:
        return self.skipped > 0


@dataclass
class ResumeResult:
    value: Any
    report: ResumeReport


class _SimulatedInterrupt(BaseException):
    """Test hook: a kill that escapes normal ``except Exception`` handling (see ``_kill_after``)."""


def run_resumable(
    plan: DurablePlan,
    store: Any,
    *,
    resources: Any = None,
    retry: RetryPolicy | None = None,
    codec: Codec | None = None,
    error_budget: int | None = None,
    _kill_after: int | None = None,
) -> ResumeResult:
    """Run ``plan`` against ``store``, skipping already-completed tasks. See module docstring.

    ``error_budget`` stops the run once the number of dead-lettered partitions exceeds it.
    ``_kill_after`` (test-only) raises an uncatchable interrupt after that many tasks commit, to
    simulate a crash mid-run; the committed journal/objects are what a resumed run recovers from. If
    the run would finish before committing that many tasks (the plan has too few partitions), it
    raises ``ValueError`` instead of completing silently — a simulated crash that can never fire is
    a misconfiguration, not an uninterrupted run.
    """
    codec = codec or PickleCodec()
    process = plan.process.resolve()
    combine = plan.combine.resolve()
    empty = plan.empty.resolve()
    if error_budget is None:
        eb = plan.stopping.get("error_budget")
        error_budget = int(eb) if eb is not None else None

    completed = store.completed()
    report = ResumeReport()
    committed = 0  # tasks committed (executed) during THIS invocation, for the kill simulation
    partials: list[tuple[int, Any]] = []  # (task index, partial) — reduced in deterministic order

    for idx, part in enumerate(plan.partitions):
        tid = plan.task_id(part)
        entry = completed.get(tid)
        if entry is not None:
            blob = store.get(entry.blob)
            if blob is not None:
                partials.append((idx, codec.decode(blob)))
                report.skipped += 1
                continue

        try:
            value = _run_one(part, process=process, combine=combine, resources=resources, retry=retry)
        except _SimulatedInterrupt:
            raise
        except BaseException as exc:
            store.record_dead(dead_letter_descriptor(tid, part, exc))
            report.dead += 1
            if error_budget is not None and report.dead > error_budget:
                report.stopped = "error_budget"
                break
            continue

        blob = store.put(codec.encode(value))
        store.record_done(tid, _partition_tag(part), blob)
        partials.append((idx, value))
        report.executed += 1
        committed += 1
        if _kill_after is not None and committed >= _kill_after:
            raise _SimulatedInterrupt(f"simulated kill after {committed} committed tasks")

    # the run finished without the kill firing: a crash requested after more commits than the plan
    # can ever produce is a misconfiguration, not a quietly-successful uninterrupted run.
    if _kill_after is not None and report.stopped is None and committed < _kill_after:
        raise ValueError(
            f"_kill_after={_kill_after} never fired: the run committed only {committed} task(s) "
            f"before completing, so the simulated crash could not happen. The plan has too few "
            f"committable partitions for that kill point."
        )

    report.dead_letters = store.dead_letters()
    value = _reduce_partials(partials, combine, empty)
    return ResumeResult(value=value, report=report)


def _run_one(
    part: Partition,
    *,
    process: Callable[..., Any],
    combine: Callable[..., Any],
    resources: Any,
    retry: RetryPolicy | None,
) -> Any:
    try:
        return process(part, resources)
    except Exception as exc:
        if retry is None:
            raise
        return retry.recover(part, exc, process=process, combine=combine, resources=resources)


def _reduce_partials(
    partials: list[tuple[int, Any]], combine: Callable[..., Any], empty: Callable[[], Any]
) -> Any:
    if not partials:
        return empty()
    # deterministic order: by task index, so an interrupted+resumed run reduces in the same order
    # as an uninterrupted run -> bit-for-bit identical (combine is associative + commutative)
    ordered = [v for _, v in sorted(partials, key=lambda kv: kv[0])]
    return reduce(combine, ordered)


def _partition_tag(p: Partition) -> str:
    return f"{p.uri}@{p.entry_start}:{p.entry_stop}"
