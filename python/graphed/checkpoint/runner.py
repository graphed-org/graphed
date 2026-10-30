"""The resumable runner (plan M8): execute a ``DurablePlan`` against a checkpoint ``Store``.

Correctness model (why resume is safe):

- A task's identity is the plan's **content-addressed** ``task_id`` (SHA-256 over the IR + process
  spec + partition). Before running a partition the runner checks the Store; if its ``task_id`` is
  already journaled with a present blob, the expensive ``process`` is **skipped** and the stored
  partial is reused. So a resumed run does **measurably less work** (``skipped`` is logged).
- The final reduction recombines the **per-task partials** (in deterministic task order) — never a
  persisted running accumulator. Each partition therefore contributes **exactly once** regardless
  of where a crash happened: **no double-count, no lost partition**, and the result is bit-for-bit
  equal to an uninterrupted run.
- A failed partition is recovered by the ``retry`` policy or harvested into the Store's dead-letter
  set with a reproducible descriptor; an **error budget** is a stopping condition.

The runner itself is single-machine; the store it takes may be local (``Store``) or at a URL
(``FsspecStore``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from graphed.core import DurablePlan, DurablePlanV2, Partition
from graphed.services import require_bound

from .codec import Codec, PickleCodec
from .errors import dead_letter_descriptor
from .resume import (
    _partition_tag,
    check_environment,
    require_environment_store,
    warn_unreused,
)
from .retry import RetryPolicy
from .store import CheckpointStore


@dataclass
class ResumeReport:
    """What a resumable run did.

    The audit trail for "measurably less work" and error harvesting.
    """

    executed: int = 0  # partitions whose process actually ran this invocation
    skipped: int = 0  # partitions reused from the checkpoint store
    dead: int = 0  # partitions harvested into the dead-letter set
    stopped: str | None = (
        None  # a StopReason value if a stopping condition fired (e.g. error budget)
    )
    dead_letters: list[dict[str, Any]] = field(default_factory=list)

    @property
    def did_less_work(self) -> bool:
        """Return whether the resumed run skipped any task."""
        return self.skipped > 0


@dataclass
class ResumeResult:
    """A resumable run's value and its report."""

    value: Any
    report: ResumeReport


class _SimulatedInterrupt(BaseException):
    """Test hook: a kill that escapes normal ``except Exception`` handling.

    See ``_kill_after``.
    """


def run_resumable(
    plan: DurablePlan,
    store: CheckpointStore,
    *,
    resources: Any = None,
    retry: RetryPolicy | None = None,
    codec: Codec | None = None,
    error_budget: int | None = None,
    accept_environment: bool = False,
    _kill_after: int | None = None,
) -> ResumeResult:
    """Run ``plan`` against ``store``, skipping already-completed tasks. See module docstring.

    ``error_budget`` stops the run once the number of dead-lettered partitions exceeds it.
    ``_kill_after`` (test-only) raises an uncatchable interrupt after that many tasks commit, to
    simulate a crash mid-run; the committed journal/objects are what a resumed run recovers from.
    If the run would finish before committing that many tasks (the plan has too few partitions), it
    raises ``ValueError`` instead of completing silently — a simulated crash that can never fire is
    a misconfiguration, not an uninterrupted run.

    The store's environment record is checked as :func:`graphed.checkpoint.resumable` checks it
    (salt ``""``), with ``accept_environment``.
    """
    require_environment_store(store)
    codec = codec or PickleCodec()
    process = plan.process.resolve()
    combine = plan.combine.resolve()
    empty = plan.empty.resolve()
    if error_budget is None:
        eb = plan.stopping.get("error_budget")
        error_budget = int(eb) if eb is not None else None

    completed = store.completed()
    check_environment(store, completed, "", accept_environment)
    report = ResumeReport()
    committed = 0  # tasks committed (executed) during THIS invocation, for the kill simulation
    # folded in task order as produced, the grouping functools.reduce gives, holding one partial
    acc: Any = None
    folded = False

    for part in plan.partitions:
        tid = plan.task_id(part)
        value = _stored_partial(store, completed.get(tid), codec)
        if value is not _NOT_STORED:
            acc, folded = (combine(acc, value) if folded else value), True
            report.skipped += 1
            continue

        try:
            value = _run_one(
                part, process=process, combine=combine, resources=resources, retry=retry
            )
        except Exception as exc:  # an interrupt or exit stops the run; it is not a dead letter
            store.record_dead(dead_letter_descriptor(tid, part, exc))
            report.dead += 1
            if error_budget is not None and report.dead > error_budget:
                report.stopped = "error_budget"
                break
            continue

        digest = store.put(codec.encode(value))
        store.record_done(tid, _partition_tag(part), digest)
        acc, folded = (combine(acc, value) if folded else value), True
        report.executed += 1
        committed += 1
        if _kill_after is not None and committed >= _kill_after:
            raise _SimulatedInterrupt(f"simulated kill after {committed} committed tasks")

    if not report.skipped:
        warn_unreused(store, completed, plan)
    _require_kill_fired(_kill_after, report, committed)
    report.dead_letters = store.dead_letters()
    value = acc if folded else empty()
    if plan.finalize is not None:
        value = plan.finalize.resolve()(value)
    return ResumeResult(value=value, report=report)


#: what :func:`_stored_partial` returns for a task with no committed partial
_NOT_STORED = object()


def _stored_partial(store: CheckpointStore, entry: Any, codec: Codec) -> Any:
    """Return the partial a journal ``entry`` committed, else :data:`_NOT_STORED`."""
    if entry is None:
        return _NOT_STORED
    blob = store.get(entry.blob)
    return _NOT_STORED if blob is None else codec.decode(blob)


def _require_kill_fired(kill_after: int | None, report: ResumeReport, committed: int) -> None:
    """Refuse a finished run whose simulated kill never fired.

    A crash requested after more commits than the plan can ever produce is a misconfiguration, not
    a quietly-successful uninterrupted run.
    """
    if kill_after is not None and report.stopped is None and committed < kill_after:
        raise ValueError(
            f"_kill_after={kill_after} never fired: the run committed only {committed} task(s) "
            f"before completing, so the simulated crash could not happen. The plan has too few "
            f"committable partitions for that kill point."
        )


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


# ---- M39: two-phase (map-write -> gather) shuffle resume ----------------------------------------
@dataclass
class ShuffleResumeResult:
    """The result of a resumable multi-stage shuffle.

    The content-addressed gather-block hashes (in dest-task order) plus an M8-style report
    (``executed``/``skipped``/``did_less_work``).
    """

    value: tuple[str, ...]
    report: ResumeReport


def run_shuffle_resumable(
    plan: DurablePlanV2,
    store: CheckpointStore,
    *,
    resources: Any = None,
    accept_environment: bool = False,
    _kill_after: int | None = None,
) -> ShuffleResumeResult:
    """Run a two-phase :class:`~graphed.core.DurablePlanV2` against ``store``, resumably.

    The phases are map-write -> gather; already-journaled blocks are skipped (plan §5.3/§7.3, the
    M8 kill/resume pattern extended to two stages). Each block is content-addressed by its V2
    ``task_id`` and journaled with its ``stage`` and ``deps``, the hash of one blob listing its
    upstream input block hashes (shared by the stage's records, so a record stays constant-size); a
    stage's tasks receive the payloads of the stages it depends on as ``inputs``. A crash at any
    point resumes from the last durable block, and the result (the tuple of gather-block hashes) is
    byte-identical to an uninterrupted run.

    Stage-process convention: ``process(task, inputs, resources) -> bytes`` where ``inputs`` is the
    tuple of upstream dep block payloads (empty for stage 0). A plan with unbound ``services`` is
    refused before its first task (``graphed.services.require_bound``). The store's environment
    record is checked as :func:`run_resumable` checks it.
    """
    require_environment_store(store)
    require_bound(plan)
    completed = store.completed()
    check_environment(store, completed, "", accept_environment)
    report = ResumeReport()
    committed = 0
    # stage index -> its blocks' payloads (in task order)
    stage_payloads: dict[int, list[bytes]] = {}
    stage_hashes: dict[int, list[str]] = {}  # stage index -> its blocks' content hashes

    for si, stage in enumerate(plan.stages):
        process = stage.process.resolve()
        upstream_payloads = [p for dep in stage.inputs for p in stage_payloads[dep]]
        upstream_hashes = tuple(h for dep in stage.inputs for h in stage_hashes[dep])
        deps: tuple[str, ...] | None = None
        this_payloads: list[bytes] = []
        this_hashes: list[str] = []
        for task in stage.tasks:
            tid = plan.task_id(si, task)
            entry = completed.get(tid)
            if entry is not None:
                blob = store.get(entry.blob)
                if blob is not None:
                    this_payloads.append(blob)
                    this_hashes.append(entry.blob)
                    report.skipped += 1
                    continue

            payload = process(task, tuple(upstream_payloads), resources)
            blob_hash = store.put(payload)
            if deps is None:
                listing = json.dumps(upstream_hashes, separators=(",", ":")).encode()
                deps = (store.put(listing),) if upstream_hashes else ()
            store.record_done(
                tid, _partition_tag(task.partition), blob_hash, stage=stage.kind, deps=deps
            )
            this_payloads.append(payload)
            this_hashes.append(blob_hash)
            report.executed += 1
            committed += 1
            if _kill_after is not None and committed >= _kill_after:
                raise _SimulatedInterrupt(f"simulated kill after {committed} committed blocks")

        stage_payloads[si] = this_payloads
        stage_hashes[si] = this_hashes

    if not report.skipped:
        warn_unreused(store, completed, plan)
    # the gather stage's content-addressed blocks
    value = tuple(stage_hashes[len(plan.stages) - 1])
    return ShuffleResumeResult(value=value, report=report)
