"""The user-facing :class:`Dashboard` (M37): start it, point an executor at ``dash.monitor``, open
``dash.url``. It bundles a local :class:`DashboardServer` with a loopback :class:`NetworkMonitor`, so
even a same-machine run streams over a real websocket (the network transport is always exercised).

Opt-in and passive (unlike dask's always-on Bokeh): nothing runs until ``start()``/``with``, and the
monitor only observes — it never changes the run's result. ``control=True`` is the exception you ask
for: the page's pause and cancel buttons then steer the attached run, and a cancel changes its result.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from graphed.core.execution import RunControl

from ._client import NetworkMonitor
from ._server import DashboardServer

if TYPE_CHECKING:
    from graphed.core.execution import Executor


class Dashboard:
    """A live execution dashboard. Example::

        with Dashboard(port=8888, profile=True) as dash:
            ProcessExecutor(monitor=dash.monitor).run(plan)   # or dash.attach(executor)
        # browse dash.url; server + client torn down on exit

    For a **remote** run, start a :class:`DashboardServer` on one host and give the executor a
    :class:`NetworkMonitor` pointed at its ``ingest_url`` on another.

    With ``control=True`` the page gains pause, resume and cancel buttons wired to :attr:`control`,
    which :meth:`attach` hands to the executor; the watching is still passive, the steering is not.
    """

    def __init__(
        self, port: int = 0, host: str = "127.0.0.1", *, profile: bool = False, control: bool = False
    ) -> None:
        self._server = DashboardServer(host, port, control=control)
        self._control = RunControl() if control else None
        self._profile = profile
        self._monitor: NetworkMonitor | None = None

    def start(self) -> Dashboard:
        if self._monitor is None:
            self._server.start()
            self._monitor = NetworkMonitor(
                self._server.ingest_url, profile=self._profile, control=self._control
            ).start()
        return self

    def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None
        self._server.stop()

    @property
    def monitor(self) -> NetworkMonitor:
        if self._monitor is None:
            raise RuntimeError("Dashboard.start() (or `with Dashboard()`) before using .monitor")
        return self._monitor

    @property
    def control(self) -> RunControl:
        if self._control is None:
            raise RuntimeError(
                "this dashboard only watches; build it with Dashboard(control=True) to steer a run"
            )
        return self._control

    @property
    def url(self) -> str:
        return self._server.url

    @property
    def server(self) -> DashboardServer:
        return self._server

    def snapshot(self) -> dict[str, Any]:
        return self._server.snapshot()

    def wait_for(self, *, finished: int, timeout: float = 10.0) -> dict[str, Any]:
        """Poll until ``finished`` tasks have landed server-side (events traverse the websocket
        asynchronously), then return the snapshot. Raises on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = self._server.snapshot()
            if snap["stats"]["finished"] >= finished:
                return snap
            time.sleep(0.02)
        raise TimeoutError(f"only {self._server.snapshot()['stats']['finished']}/{finished} finished")

    def attach(self, executor: Executor) -> Executor:
        """Set ``executor.monitor`` (and ``executor.control`` with ``control=True``). Refuses an
        executor without those attributes, since a new attribute would be one nothing reads."""
        names = ("monitor", "control") if self._control is not None else ("monitor",)
        missing = [n for n in names if not hasattr(executor, n)]
        if missing:
            raise TypeError(
                f"{type(executor).__name__} takes no {' or '.join(missing)}; Dashboard.attach cannot wire it"
            )
        executor.monitor = self.monitor  # type: ignore[attr-defined]
        if self._control is not None:
            executor.control = self._control  # type: ignore[attr-defined]
        return executor

    def __enter__(self) -> Dashboard:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
