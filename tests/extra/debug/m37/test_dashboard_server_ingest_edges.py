"""The dashboard server drops a message it cannot decode instead of raising on its IOLoop."""

from __future__ import annotations

import base64
import json

import pytest

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


def _profile(tree: object) -> dict[str, str]:
    return {"tree_b64": base64.b64encode(json.dumps(tree).encode()).decode()}


@pytest.mark.parametrize(
    "tree",
    [
        [],
        {"count": 1},
        {"count": 1, "children": []},
        {"count": True, "children": {}},
        {"count": 1, "children": {"f": {"count": 1, "children": {}}}},
    ],
)
def test_a_profile_that_is_json_but_not_a_count_tree_is_dropped(tree: object) -> None:
    server = DashboardServer()
    before = (server.flamegraph(), server.snapshot())
    server._ingest_profile(_profile(tree))
    assert (server.flamegraph(), server.snapshot()) == before


def test_a_count_tree_still_merges() -> None:
    server = DashboardServer()
    leaf = {"name": "f", "count": 2, "children": {}}
    server._ingest_profile(_profile({"name": "all", "count": 3, "children": {"f;a.py;1": leaf}}))
    assert server.flamegraph()["children"] == [{"name": "f", "value": 2, "children": []}]
    assert server.snapshot()["profile_samples"] == 3
