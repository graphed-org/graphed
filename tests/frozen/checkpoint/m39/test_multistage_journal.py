"""M39 — the multi-stage journal schema + journal-per-writer (plan §7.3, §6.3).

The flat single-phase journal cannot represent a phase-2 task depending on a phase-1 output. M39 adds
``stage`` and ``deps`` (the input block hashes for a gather-block) to ``JournalEntry`` so resume knows
which map-write and gather blocks exist. On a shared FS, N nodes appending to one ``journal.log`` is a
contention hazard, so each writer journals its OWN ``journal.<node>.log`` and resume replays the UNION;
the owner journals a stolen task's manifest under its own writer file (so a completed stolen task is
skipped on resume, not needlessly re-run). The V1 default (no ``node``) still writes ``journal.log``.
"""

from __future__ import annotations

from graphed.checkpoint import Store


def test_journal_entry_carries_stage_and_deps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = Store(tmp_path)
    block = store.put(b"gather-block")
    dep_a = store.put(b"map-write-a")
    dep_b = store.put(b"map-write-b")
    store.record_done("gj-0", "dest@p0", block, stage="gather_join", deps=(dep_a, dep_b))
    entry = store.completed()["gj-0"]
    assert entry.stage == "gather_join", "the journal must record which stage a block belongs to"
    assert tuple(entry.deps) == (dep_a, dep_b), "a gather block records its input block hashes as deps"


def test_default_writer_still_uses_the_v1_journal_file(tmp_path) -> None:
    # backward-compat: with no node the journal file is journal.log (M8 tests are untouched).
    store = Store(tmp_path)
    store.record_done("t0", "src@0:1", store.put(b"x"), stage="map_write")
    assert (tmp_path / "journal.log").exists()


def test_journal_per_writer_and_union_replay(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # each node writes its OWN journal file; a reader replays the UNION (no shared-FS append contention).
    node_a = Store(tmp_path, node="A")
    node_b = Store(tmp_path, node="B")
    ha = node_a.put(b"block-from-A")
    hb = node_b.put(b"block-from-B")
    node_a.record_done("ta", "src@0:1", ha, stage="map_write")
    node_b.record_done("tb", "src@1:2", hb, stage="map_write")
    assert (tmp_path / "journal.A.log").exists()
    assert (tmp_path / "journal.B.log").exists()

    union = Store(tmp_path).completed()
    assert set(union) == {"ta", "tb"}, "resume replays the UNION of every node's journal"


def test_owner_journals_a_stolen_tasks_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # a stolen producer-task's block stays on the thief; its owner journals the tiny MANIFEST so the
    # completed stolen task is SKIPPED (not re-run) on resume. Attribution is unambiguous per-writer.
    thief = Store(tmp_path, node="thief")
    owner = Store(tmp_path, node="owner")
    block = thief.put(b"stolen-producer-block")  # block lives in the thief's store
    manifest = owner.put(b'{"dest0": ["' + block.encode() + b'"]}')
    owner.record_done("prod-7", "src@7:8", manifest, stage="manifest", deps=(block,))

    entry = Store(tmp_path).completed()["prod-7"]
    assert entry.stage == "manifest"
    assert block in entry.deps, "the manifest names the block it points at (held on the thief)"
