"""The dashboard server drops a message that is not JSON instead of raising on its IOLoop."""

from __future__ import annotations

from graphed.debug import DashboardServer


def test_a_message_that_is_not_json_is_dropped() -> None:
    server = DashboardServer()
    before = server.snapshot()
    server._ingest("{not json")
    assert server.snapshot() == before
