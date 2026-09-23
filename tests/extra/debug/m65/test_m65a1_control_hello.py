"""Only a ``hello`` with ``control: true`` makes an ``/ingest`` connection a control listener."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

import pytest

pytest.importorskip("perspective")
pytest.importorskip("tornado")
websocket = pytest.importorskip("websocket")

from graphed.debug import DashboardServer  # noqa: E402


@pytest.mark.parametrize(
    ("hello", "listeners"),
    [
        ({"type": "hello", "control": False}, 0),
        ({"type": "hello", "lean": True}, 0),
        ({"type": "hello", "control": True}, 1),
    ],
)
def test_only_a_control_hello_makes_a_listener(hello: dict[str, Any], listeners: int) -> None:
    server = DashboardServer(control=True).start()
    try:
        conn = websocket.create_connection(server.ingest_url, timeout=5)
        conn.send(json.dumps(hello))
        conn.send(json.dumps({"type": "combine"}))  # ingested after the hello: a sync point
        deadline = time.monotonic() + 5.0
        while server.snapshot()["stats"]["combines"] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.snapshot()["stats"]["combines"] == 1
        assert server.snapshot()["control_listeners"] == listeners
        req = urllib.request.Request(
            server.url + "api/control",
            data=json.dumps({"cmd": "pause"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert json.loads(resp.read()) == {"cmd": "pause", "delivered": listeners}
        conn.close()
    finally:
        server.stop()


def test_a_failed_hello_closes_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    from graphed.core import RunControl  # noqa: PLC0415
    from graphed.debug import NetworkMonitor  # noqa: PLC0415

    class FakeConn:
        closed = False

        def send(self, msg: str) -> None:
            raise ConnectionError("boom")

        def close(self) -> None:
            self.closed = True

    conn = FakeConn()
    monkeypatch.setattr(websocket, "create_connection", lambda url, timeout=5: conn)
    mon = NetworkMonitor("ws://fake/ingest", control=RunControl())
    with pytest.raises(ConnectionError):
        mon._connect()
    assert conn.closed is True
