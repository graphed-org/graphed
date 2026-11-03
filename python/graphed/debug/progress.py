"""A progress bar for a run, as a monitor any runner takes."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

from graphed.core import MonitorBase, TaskEvent, TaskPhase


class _TextBar:
    """The part of ``tqdm`` :class:`ProgressBar` drives, as one rewritten line on stderr."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0
        self._shown = 0.0

    def update(self) -> None:
        self.n += 1
        if time.monotonic() - self._shown >= 0.1:  # a run may have 100k tasks
            self.refresh()

    def refresh(self) -> None:
        self._shown = time.monotonic()
        print(f"\rtasks: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        self.refresh()
        print(file=sys.stderr)


def _open_bar(total: int) -> Any:
    try:
        from tqdm.auto import tqdm  # here, as every worker imports this module
    except ImportError:
        return _TextBar(total)
    return tqdm(total=total, unit="task")


class ProgressBar(MonitorBase):
    """One tick per finished task: ``ProcessPoolExecutor(monitor=ProgressBar()).run(plan)``.

    Uses ``tqdm`` when it is installed and a plain text line otherwise. It closes when the run
    ends, however it ends, and one bar can watch several runs in turn. The total is the plan's task
    count and follows an adaptive plan as its tasks arrive.

    A staged plan (a join or a repartition) emits no task events, so its bar shows no tasks.
    """

    lean_events = True
    complete_events = True
    _submitted: int

    def __init__(self) -> None:
        """Make a bar; it appears when a run starts."""
        self._lock = threading.Lock()
        self._bar: Any = None

    def on_run_start(self, n_tasks: int) -> None:
        """Open the bar."""
        with self._lock:
            self._submitted = 0
            self._bar = _open_bar(n_tasks)

    def on_task(self, event: TaskEvent) -> None:
        """Tick for a finished task, and grow the total past the plan's own tasks."""
        with self._lock:
            if self._bar is None:
                return
            if event.phase is TaskPhase.SUBMITTED:
                self._submitted += 1
                self._bar.total = max(self._bar.total, self._submitted)  # the next tick draws it
            elif event.phase is TaskPhase.FINISHED:
                self._bar.update()

    def on_run_end(self) -> None:
        """Close the bar."""
        with self._lock:
            bar, self._bar = self._bar, None
        if bar is not None:
            bar.close()
