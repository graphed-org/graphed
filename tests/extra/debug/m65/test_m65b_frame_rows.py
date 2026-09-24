"""The server writes a frame's task rows in one Perspective update, not one per item."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
websocket = pytest.importorskip("websocket")

import graphed.debug.dashboard._wire as wire  # noqa: E402
from graphed.core import TaskEvent, TaskPhase  # noqa: E402
from graphed.debug import DashboardServer  # noqa: E402

N = 100


class _Counting:
    def __init__(self, table: Any) -> None:
        self.table = table
        self.sizes: list[int] = []

    def update(self, rows: list[dict[str, Any]]) -> None:
        self.sizes.append(len(rows))
        self.table.update(rows)

    def view(self) -> Any:
        return self.table.view()


def _frame(events: list[TaskEvent]) -> str:
    return json.dumps({"type": "batch", "items": [wire.task_message(e) for e in events]})


def _wait(server: DashboardServer, stat: str, n: int) -> None:
    deadline = time.monotonic() + 5.0
    while server.snapshot()["stats"][stat] < n and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.snapshot()["stats"][stat] == n


def test_a_frame_is_one_update() -> None:
    server = DashboardServer().start()
    try:
        counting = _Counting(server._tasks)
        server._tasks = counting
        conn = websocket.create_connection(server.ingest_url, timeout=5)
        try:
            sub = TaskPhase.SUBMITTED
            conn.send(_frame([TaskEvent(sub, k, "driver", 0.0, f"L{k}", k) for k in range(N)]))
            _wait(server, "submitted", N)
            # label-less terminals beside labelled submits: the terminals keep their labels
            fin = [TaskEvent(TaskPhase.FINISHED, k, "w0", 1.0, "", 0) for k in range(N)]
            conn.send(_frame(fin + [TaskEvent(sub, k, "driver", 0.0, f"L{k}", k) for k in range(N, 2 * N)]))
            _wait(server, "finished", N)
        finally:
            conn.close()
        assert counting.sizes == [N, 2 * N]
        rows = {r["key"]: r for r in counting.view().to_records()}
        assert all((rows[k]["phase"], rows[k]["partition"]) == ("finished", f"L{k}") for k in range(N))
        assert all(
            (rows[k]["phase"], rows[k]["partition"]) == ("submitted", f"L{k}") for k in range(N, 2 * N)
        )
    finally:
        server.stop()
