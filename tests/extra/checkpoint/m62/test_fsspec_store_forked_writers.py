"""Forked copies of one ``FsspecStore`` keep every record, as the local ``Store`` does."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from graphed.checkpoint import FsspecStore

FORK = [m for m in multiprocessing.get_all_start_methods() if m == "fork"]


def _child(store: FsspecStore, w: int) -> None:
    for i in range(3):
        store.record_dead({"task_id": f"w{w}-{i}", "error_type": "E"})
    store.record_done(f"t{w}", "p", store.put(f"b{w}".encode()))


@pytest.mark.parametrize("method", FORK)
def test_forked_children_do_not_overwrite_records(tmp_path: Path, method: str) -> None:
    assert method == "fork"
    url = (tmp_path / "s").as_uri()
    store = FsspecStore(url)
    store.record_dead({"task_id": "parent-0", "error_type": "E"})
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_child, args=(store, w)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    store.record_dead({"task_id": "parent-1", "error_type": "E"})

    fresh = FsspecStore(url)
    dead = sorted(str(d["task_id"]) for d in fresh.dead_letters())
    expected = ["parent-0", "parent-1"] + [f"w{w}-{i}" for w in range(4) for i in range(3)]
    assert dead == sorted(expected)
    assert sorted(fresh.completed()) == ["t0", "t1", "t2", "t3"]
