"""An event emitted while the sender has no connection yet still reaches the server."""

from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
pytest.importorskip("websocket")

from graphed.core import TaskEvent, TaskPhase
from graphed.debug import DashboardServer, NetworkMonitor


def test_an_event_emitted_just_before_the_first_pop_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    real = NetworkMonitor._drain
    injected: list[NetworkMonitor] = []

    def drain(self: NetworkMonitor) -> list[dict[str, Any]]:
        if not injected:  # the run thread's first event lands just before the sender's first pop
            injected.append(self)
            self.on_task(TaskEvent(TaskPhase.FINISHED, 0, "w0", 1.0, ""))
        return real(self)

    monkeypatch.setattr(NetworkMonitor, "_drain", drain)
    server = DashboardServer().start()
    try:
        mon = NetworkMonitor(server.ingest_url).start()
        mon.close()
        assert injected == [mon]
        deadline = time.monotonic() + 5.0
        while server.snapshot()["stats"]["finished"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.snapshot()["stats"]["finished"] == 1
        assert server.snapshot()["ingest_connections"] == 1
    finally:
        server.stop()
