"""The wire contract shared by the dashboard client (executor side) and server (dashboard side).

Events travel as JSON text over a websocket. The server materialises them into three Perspective
tables whose schemas live here so client and server never drift.
"""

from __future__ import annotations

from typing import Any

from graphed_core.execution import TaskEvent

# Perspective table schemas (column -> perspective type). ``tasks`` is indexed by ``key`` so a task's
# row advances SUBMITTED -> STARTED -> FINISHED/ERRORED in place; ``stats`` is a single indexed row
# updated in place. (The profile is a flamegraph, not a table — the server merges sampled stack trees
# and serves them at /api/flamegraph.json.)
TASKS_SCHEMA: dict[str, str] = {
    "key": "integer",
    "phase": "string",
    "worker": "string",
    "t": "float",
    "partition": "string",
    "n_entries": "integer",
    "error": "string",
}
STATS_SCHEMA: dict[str, str] = {
    "metric": "string",
    "submitted": "integer",
    "started": "integer",
    "finished": "integer",
    "errored": "integer",
    "inflight": "integer",
    "combines": "integer",
}

STATS_KEYS = ("submitted", "started", "finished", "errored", "inflight", "combines")


def task_message(event: TaskEvent) -> dict[str, Any]:
    return {
        "type": "task",
        "key": event.key,
        "phase": event.phase.value,
        "worker": event.worker,
        "t": event.t,
        "partition": event.partition,
        "n_entries": event.n_entries,
        "error": event.error or "",
    }


def combine_message(leaves_done: int) -> dict[str, Any]:
    return {"type": "combine", "leaves_done": leaves_done}


def profile_message(worker: str, tree_b64: str) -> dict[str, Any]:
    return {"type": "profile", "worker": worker, "tree_b64": tree_b64}


def task_row(message: dict[str, Any]) -> dict[str, Any]:
    """The Perspective ``tasks``-table row for a decoded task message."""
    return {col: message.get(col) for col in TASKS_SCHEMA}
