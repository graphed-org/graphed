"""The dashboard server drops a message it cannot decode instead of raising on its IOLoop."""

from __future__ import annotations

from graphed.debug import DashboardServer


def test_a_message_that_is_not_json_is_dropped() -> None:
    server = DashboardServer()
    before = server.snapshot()
    server._ingest("{not json")
    assert server.snapshot() == before


def test_a_message_nested_past_the_recursion_limit_is_dropped() -> None:
    server = DashboardServer()
    before = server.snapshot()
    server._ingest("[" * 100_000 + "]" * 100_000)
    assert server.snapshot() == before


def test_a_json_message_that_is_not_an_object_is_dropped() -> None:
    server = DashboardServer()
    before = server.snapshot()
    server._ingest("[1, 2]")
    assert server.snapshot() == before
