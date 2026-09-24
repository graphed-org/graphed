"""m62 unit B — ``FsspecStore`` blobs and records on ``memory://`` and ``file://``."""

from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import m62_url_helpers as h
import pytest

from graphed.checkpoint import FsspecStore, JournalEntry, Store


def test_put_is_content_addressed_and_idempotent(store_url) -> None:
    store = FsspecStore(store_url)
    h1 = store.put(b"hello")
    h2 = store.put(b"hello")
    assert h1 == h2 == Store.content_hash(b"hello")
    assert [p.rsplit("/", 1)[-1] for p in store.fs.find(store.objects)] == [h1]
    assert store.get(h1) == b"hello"
    assert store.put(b"world") != h1
    assert len(store.fs.find(store.objects)) == 2


def test_get_missing_blob_is_none(store_url) -> None:
    store = FsspecStore(store_url)
    assert store.get("0" * 64) is None
    store.put(b"present")
    assert store.get("0" * 64) is None


def test_fresh_store_is_empty(store_url) -> None:
    store = FsspecStore(store_url)
    assert store.completed() == {}
    assert store.dead_letters() == []


def test_journal_replays_stage_deps_and_last_record_wins(store_url) -> None:
    store = FsspecStore(store_url)
    blobs = [store.put(f"partial-{i}".encode()) for i in range(12)]
    for i, blob in enumerate(blobs):
        store.record_done("task-A", f"uri@{i}:{i + 1}", blob, stage=f"s{i}", deps=(f"d{i}", "x"))
    other = store.put(b"partial-B")
    store.record_done("task-B", "uri@20:30", other)
    done = FsspecStore(store_url).completed()
    assert done == {
        "task-A": JournalEntry("task-A", "uri@11:12", blobs[11], "s11", ("d11", "x")),
        "task-B": JournalEntry("task-B", "uri@20:30", other, "", ()),
    }


def test_record_without_its_blob_is_not_honored(store_url) -> None:
    store = FsspecStore(store_url)
    store.record_done("task-X", "uri@0:10", "f" * 64)
    assert store.completed() == {}
    blob = store.put(b"partial-A")
    store.record_done("task-A", "uri@0:10", blob)
    assert set(store.completed()) == {"task-A"}


def test_unparseable_record_is_skipped(store_url) -> None:
    store = FsspecStore(store_url)
    blob = store.put(b"partial-A")
    store.record_done("task-A", "uri@0:10", blob)
    store.fs.pipe_file(f"{store.journal_path}/torn", b'{"task_id": "task-B", "parti')
    assert len(store.fs.find(store.journal_path)) == 2
    assert set(store.completed()) == {"task-A"}


def _feed(store) -> None:
    blob = store.put(b"partial-A")
    store.record_done("task-A", "uri@0:10", blob, stage="map_write", deps=("d0", "d1"))
    store.record_done("task-B", "uri@10:20", store.put(b"partial-B"))
    store.record_dead({"task_id": "t1", "error_type": "ValueError", "detail": {"b": 2, "a": [1, "x"]}})
    store.record_dead({"task_id": "t2", "error_type": "MemoryError"})


def test_records_are_one_line_objects_with_the_local_bytes(store_url, tmp_path) -> None:
    local_dead: list[bytes] = []
    for node in (None, "A"):
        remote = FsspecStore(store_url, node=node)
        local = Store(tmp_path / f"local-{node}", node=node)
        _feed(remote)
        _feed(local)
        journal = h.record_objects(remote, remote.journal_path)
        dead = h.record_objects(remote, remote.dead_letter_path)
        for record in journal + dead:
            assert record.count(b"\n") == 1
            assert record.endswith(b"\n")
        assert len(journal) == 2
        assert sorted(journal) == sorted(h.local_lines(Path(local.journal_path)))
        local_dead += h.local_lines(Path(local.dead_letter_path))
        assert sorted(dead) == sorted(local_dead)


def test_dead_letters_keep_insertion_order(store_url) -> None:
    store = FsspecStore(store_url)
    for i in range(13):
        store.record_dead({"task_id": f"t{i}", "error_type": "ValueError"})
    assert [d["task_id"] for d in store.dead_letters()] == [f"t{i}" for i in range(13)]
    assert [d["task_id"] for d in FsspecStore(store_url).dead_letters()] == [f"t{i}" for i in range(13)]


def test_node_writers_replay_as_a_union(store_url) -> None:
    a = FsspecStore(store_url, node="A")
    b = FsspecStore(store_url, node="B")
    a.record_done("task-A", "uri@0:10", a.put(b"partial-A"))
    b.record_done("task-B", "uri@10:20", b.put(b"partial-B"))
    reader = FsspecStore(store_url)
    assert len(reader.fs.find(a.journal_path)) == 1
    assert len(reader.fs.find(b.journal_path)) == 1
    assert reader.fs.find(reader.journal_path) == []
    assert set(reader.completed()) == {"task-A", "task-B"}


def test_concurrent_record_dead_loses_nothing(store_url) -> None:
    store = FsspecStore(store_url)
    barrier = threading.Barrier(8)

    def write(worker: int) -> None:
        barrier.wait()
        for i in range(5):
            store.record_dead({"task_id": "same", "error_type": "RuntimeError"})
            store.record_dead({"task_id": f"w{worker}-{i}", "error_type": "ValueError"})

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(write, range(8)))
    got = Counter(json.dumps(d, sort_keys=True) for d in FsspecStore(store_url).dead_letters())
    want = Counter({json.dumps({"error_type": "RuntimeError", "task_id": "same"}, sort_keys=True): 40})
    for worker in range(8):
        for i in range(5):
            want[json.dumps({"error_type": "ValueError", "task_id": f"w{worker}-{i}"}, sort_keys=True)] = 1
    assert got == want
    assert len(store.fs.find(store.dead_letter_path)) == 80


def test_unreadable_record_raises_the_backend_error(monkeypatch) -> None:
    store = FsspecStore(f"memory://m62-{uuid.uuid4().hex}")
    store.record_done("task-A", "uri@0:10", store.put(b"partial-A"))
    store.record_done("task-B", "uri@10:20", store.put(b"partial-B"))
    store.record_dead({"task_id": "t1", "error_type": "ValueError"})
    store.record_dead({"task_id": "t2", "error_type": "MemoryError"})
    real = store.fs.cat_ranges
    boom = OSError("boom")
    calls: list[list[str]] = []

    def fake(paths, starts, ends, **kwargs):
        calls.append(list(paths))
        return [*real(paths, starts, ends, **kwargs)[:-1], boom]

    monkeypatch.setattr(store.fs, "cat_ranges", fake)
    with pytest.raises(OSError) as raised:
        store.completed()
    assert raised.value is boom
    with pytest.raises(OSError) as raised:
        store.dead_letters()
    assert raised.value is boom
    assert calls
    assert all(len(paths) == 2 for paths in calls)
