"""A run's task events folded into a report a preservation bundle can keep (plan m65 C).

``RunRecorder`` is a monitor that records every event at the driver and forwards to an optional
inner monitor (a dashboard). ``report()`` folds what arrived since the previous report into a
``RunReport``: per task its state, worker, duration and error; the run's outcome, its ``StageError``
and the environment digest a bundle would record.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import Any

from graphed.core import ExecResult, Monitor, StopReason, TaskEvent, TaskPhase, WorkerProfiler

from .errors import SourceFrame, StageError

_VERSION = 1
_TERMINAL = 2
_CLASS = {
    TaskPhase.SUBMITTED: 0,
    TaskPhase.STARTED: 1,
    TaskPhase.FINISHED: _TERMINAL,
    TaskPhase.ERRORED: _TERMINAL,
}
_FAILURE_FIELDS = (
    "op",
    "partition",
    "cause_type",
    "cause_message",
    "opt_level",
    "variation",
)


@dataclass(frozen=True)
class TaskRecord:
    """One task's latest attempt: ``state`` is ``submitted|started|finished|errored``."""

    key: int
    partition: str
    worker: str
    state: str
    duration_s: float | None
    error: str | None


@dataclass(frozen=True)
class RunReport:
    """What a run did, as plain data (JSON version 1 through ``to_json``/``from_json``)."""

    outcome: str
    n_partitions: int | None
    wall_s: float
    tasks: tuple[TaskRecord, ...]
    failure: StageError | None
    error: str | None
    environment: dict[str, Any]
    environment_digest: str
    #: service name -> the endpoint this run reached it at; run provenance, outside any fingerprint
    endpoints: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoints", MappingProxyType(dict(self.endpoints)))

    def __reduce__(self) -> tuple[Any, ...]:
        # a mappingproxy does not pickle; the constructor re-wraps the plain dict
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return (_report, (values | {"endpoints": dict(self.endpoints)},))

    def __setstate__(self, state: dict[str, Any]) -> None:  # a 0.0.6 pickle carries no endpoints
        self.__dict__.update({"endpoints": MappingProxyType({})} | state)

    @property
    def failed_keys(self) -> tuple[int, ...]:
        return tuple(t.key for t in self.tasks if t.state == "errored")

    def to_json(self) -> dict[str, Any]:
        f = self.failure
        failure = None
        if f is not None:
            failure = {name: getattr(f, name) for name in _FAILURE_FIELDS}
            failure["frames"] = [[x.filename, x.lineno, x.function, x.source] for x in f.frames]
            failure["input_forms"] = list(f.input_forms)
        return {
            "version": _VERSION,
            "outcome": self.outcome,
            "n_partitions": self.n_partitions,
            "wall_s": self.wall_s,
            "tasks": [asdict(t) for t in self.tasks],
            "failure": failure,
            "error": self.error,
            "environment": self.environment,
            "environment_digest": self.environment_digest,
        } | ({"endpoints": dict(self.endpoints)} if self.endpoints else {})

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RunReport:
        if data["version"] != _VERSION:
            raise ValueError(f"unsupported run report version {data['version']}")
        f = data["failure"]
        failure = None
        if f is not None:
            failure = StageError(
                frames=tuple(SourceFrame(*x) for x in f["frames"]),
                input_forms=tuple(f["input_forms"]),
                **{name: f[name] for name in _FAILURE_FIELDS},
            )
        return cls(
            outcome=data["outcome"],
            n_partitions=data["n_partitions"],
            wall_s=data["wall_s"],
            tasks=tuple(TaskRecord(**t) for t in data["tasks"]),
            failure=failure,
            error=data["error"],
            environment=data["environment"],
            environment_digest=data["environment_digest"],
            endpoints=data.get("endpoints", {}),
        )


def _report(values: dict[str, Any]) -> RunReport:
    return RunReport(**values)


def _fold(key: int, events: list[TaskEvent]) -> TaskRecord:
    # an event whose phase class the current lifecycle already holds starts a new one (a retry)
    life: dict[int, TaskEvent] = {}
    last = events[0]
    label = ""
    for e in events:
        c = _CLASS[e.phase]
        if c in life:
            life = {}
        life[c] = last = e
        if e.phase is TaskPhase.SUBMITTED:
            label = e.partition
    term, started = life.get(_TERMINAL), life.get(1)
    state = term.phase.value if term else "started" if started else "submitted"
    duration = term.t - started.t if term and started else None
    return TaskRecord(key, label, last.worker, state, duration, term.error if term else None)


class RunRecorder:
    """A ``Monitor`` that records a run's events for ``report()``, then forwards every call to
    ``inner``. It defines neither ``lean_events`` nor ``worker_monitor_factory``, so the executor
    routes the full stream through the driver, and it opts into ``complete_events``."""

    complete_events = True

    def __init__(self, inner: Monitor | None = None) -> None:
        self._inner = inner
        self._calls: list[tuple[TaskEvent, float]] = []

    def on_task(self, event: TaskEvent) -> None:
        self._calls.append((event, time.perf_counter()))  # before forwarding: inner may raise
        if self._inner is not None:
            self._inner.on_task(event)

    def on_profile(self, worker: str, payload: bytes) -> None:
        if self._inner is not None:
            self._inner.on_profile(worker, payload)

    def on_combine(self, leaves_done: int) -> None:
        if self._inner is not None:
            self._inner.on_combine(leaves_done)

    def worker_profiler_factory(self) -> Callable[[], WorkerProfiler] | None:
        return self._inner.worker_profiler_factory() if self._inner is not None else None

    def report(
        self,
        *,
        result: ExecResult[Any] | None = None,
        error: BaseException | None = None,
        container_digest: str | None = None,
        endpoints: Mapping[str, str] | None = None,
    ) -> RunReport:
        """Fold the events received since the previous ``report()`` (or construction) into a
        ``RunReport`` of ``result`` or of the raised ``error`` — exactly one of them; ``endpoints``
        records where the run reached each service."""
        if (result is None) == (error is None):
            raise ValueError("RunRecorder.report needs exactly one of result= or error=")
        from graphed.preserve import capture_environment, fingerprint  # noqa: PLC0415

        n = len(self._calls)  # appends racing this take land whole in the next report
        calls = self._calls[:n]
        del self._calls[:n]
        by_key: dict[int, list[TaskEvent]] = {}
        for event, _ in calls:
            by_key.setdefault(event.key, []).append(event)
        env = capture_environment(container_digest)
        outcome = "failed"
        n_partitions: int | None = None
        message: str | None = None
        if error is not None:
            message = f"{type(error).__name__}: {error}"
        elif result is not None:
            stopped = result.stopped
            outcome = "completed" if stopped in (None, StopReason.EXHAUSTED) else str(stopped)
            n_partitions = result.n_partitions
        return RunReport(
            outcome=outcome,
            n_partitions=n_partitions,
            wall_s=calls[-1][1] - calls[0][1] if calls else 0.0,
            tasks=tuple(_fold(k, by_key[k]) for k in sorted(by_key)),
            failure=error if isinstance(error, StageError) else None,
            error=message,
            environment=env,
            environment_digest=fingerprint(env),
            endpoints=endpoints or {},
        )
