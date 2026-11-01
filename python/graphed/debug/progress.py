"""A progress bar for a run, as a monitor any runner takes."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

from graphed.core import MonitorBase, TaskEvent, TaskPhase

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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


class ProgressBar(MonitorBase):
    """One tick per finished task: ``ProcessPoolExecutor(monitor=ProgressBar()).run(plan)``.

    Uses ``tqdm`` when it is installed and a plain text line otherwise. It closes when the run
    ends, however it ends, and one bar can watch several runs in turn. The total is the plan's task
    count and follows an adaptive plan as its tasks arrive.

    A staged plan (a join or a repartition) emits no task events, so it shows no progress.
    """

    lean_events = True
    complete_events = True

    def __init__(self) -> None:
        """Make a bar; it appears when a run starts."""
        self._lock = threading.Lock()
        self._bar: Any = None
        self._submitted = 0

    def on_run_start(self, n_tasks: int) -> None:
        """Open the bar."""
        with self._lock:
            self._submitted = 0
            self._bar = _TextBar(n_tasks) if tqdm is None else tqdm(total=n_tasks, unit="task")

    def on_task(self, event: TaskEvent) -> None:
        """Tick for a task that finished or failed."""
        with self._lock:
            if self._bar is None:
                return
            if event.phase is TaskPhase.SUBMITTED:
                self._submitted += 1
                if self._submitted > self._bar.total:
                    self._bar.total = self._submitted
                    self._bar.refresh()
            elif event.phase is not TaskPhase.STARTED:
                self._bar.update()

    def on_run_end(self) -> None:
        """Close the bar."""
        with self._lock:
            bar, self._bar = self._bar, None
        if bar is not None:
            bar.close()
