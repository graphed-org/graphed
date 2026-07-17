"""M8 — the content-addressed Store is crash-safe and idempotent."""

from __future__ import annotations

from graphed.checkpoint import Store


def test_put_is_content_addressed_and_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = Store(tmp_path)
    h1 = s.put(b"hello")
    h2 = s.put(b"hello")  # identical content -> same name, no second object
    assert h1 == h2 == Store.content_hash(b"hello")
    assert s.get(h1) == b"hello"
    assert s.put(b"world") != h1


def test_get_missing_blob_is_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert Store(tmp_path).get("0" * 64) is None


def test_journal_replays_completed_tasks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = Store(tmp_path)
    blob = s.put(b"partial-A")
    s.record_done("task-A", "uri@0:10", blob)
    done = s.completed()
    assert set(done) == {"task-A"}
    assert done["task-A"].blob == blob


def test_journal_entry_without_its_blob_is_not_honored(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # guards a crash where a journal line outraced its object write: do not claim it done
    s = Store(tmp_path)
    s.record_done("task-X", "uri@0:10", "f" * 64)  # blob never written
    assert s.completed() == {}


def test_torn_trailing_journal_line_is_ignored(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = Store(tmp_path)
    blob = s.put(b"partial-A")
    s.record_done("task-A", "uri@0:10", blob)
    # simulate an interrupted append (a half-written final line)
    with open(s.journal_path, "a", encoding="utf-8") as f:
        f.write('{"task_id": "task-B", "parti')
    done = s.completed()
    assert set(done) == {"task-A"}  # the torn line is dropped, recovery is not corrupted


def test_dead_letter_set_is_append_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = Store(tmp_path)
    s.record_dead({"task_id": "t1", "error_type": "ValueError"})
    s.record_dead({"task_id": "t2", "error_type": "MemoryError"})
    dl = s.dead_letters()
    assert [d["task_id"] for d in dl] == ["t1", "t2"]
