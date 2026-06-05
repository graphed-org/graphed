"""The content-addressed checkpoint Store (plan M8).

A local-filesystem store (the MVP guardrail: **no distributed store**) with three durable parts:

- **objects/** — content-addressed blobs. ``put`` writes a blob named by its SHA-256, *atomically*
  (write to a temp file in the same directory, ``fsync``, then ``rename``), so an interrupted write
  never leaves a torn object visible. Writes are idempotent: the same content always maps to the
  same name, so re-running a task is a no-op (cache-poisoning-safe — the name *is* the hash).
- **journal.log** — an append-only manifest of completed tasks (one JSON line per task, ``fsync``'d).
  Resume replays it to learn what is already done. A torn trailing line (a crash mid-append) is
  ignored on replay, so a half-written journal never corrupts recovery.
- **dead_letter.log** — an append-only set of failures (the harvested ``StageError`` descriptor +
  partition + provenance), so a poison partition is recorded reproducibly rather than lost.

The Store is what makes resume correct: the resumable runner consults ``completed`` to **skip work
already done** and recombines per-task outputs (never a persisted running accumulator), so a crash
at any point causes **no double-count and no lost partition**.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JournalEntry:
    """One completed task recorded in the manifest."""

    task_id: str
    partition: str  # a human-readable partition tag (uri@start:stop), for audit
    blob: str  # content hash of the stored output


class Store:
    """A content-addressed, append-only, crash-safe checkpoint store on the local filesystem."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.journal_path = self.root / "journal.log"
        self.dead_letter_path = self.root / "dead_letter.log"
        self.objects.mkdir(parents=True, exist_ok=True)

    # ---- content-addressed blobs ----------------------------------------------------------------
    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def put(self, data: bytes) -> str:
        """Store ``data`` under its content hash, atomically and idempotently. Returns the hash."""
        digest = self.content_hash(data)
        dest = self.objects / digest
        if dest.exists():  # idempotent: identical content is already committed
            return digest
        self._atomic_write(dest, data)
        return digest

    def has_blob(self, digest: str) -> bool:
        return (self.objects / digest).exists()

    def get(self, digest: str) -> bytes | None:
        path = self.objects / digest
        return path.read_bytes() if path.exists() else None

    # ---- append-only manifest / journal ---------------------------------------------------------
    def record_done(self, task_id: str, partition: str, blob: str) -> None:
        self._append(self.journal_path, {"task_id": task_id, "partition": partition, "blob": blob})

    def completed(self) -> dict[str, JournalEntry]:
        """Replay the journal into ``task_id -> JournalEntry`` (last write wins). A torn trailing
        line (interrupted append) is skipped, never fatal."""
        done: dict[str, JournalEntry] = {}
        for rec in self._read_lines(self.journal_path):
            blob = rec.get("blob")
            # only honor an entry whose blob is actually present (guards a journal line that
            # outraced its object write across a crash)
            if isinstance(blob, str) and self.has_blob(blob):
                tid = str(rec.get("task_id", ""))
                done[tid] = JournalEntry(tid, str(rec.get("partition", "")), blob)
        return done

    # ---- dead-letter set ------------------------------------------------------------------------
    def record_dead(self, descriptor: Mapping[str, object]) -> None:
        self._append(self.dead_letter_path, dict(descriptor))

    def dead_letters(self) -> list[dict[str, object]]:
        return list(self._read_lines(self.dead_letter_path))

    # ---- internals ------------------------------------------------------------------------------
    @staticmethod
    def _atomic_write(dest: Path, data: bytes) -> None:
        # temp file in the SAME directory so rename is atomic on the same filesystem
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)  # atomic on POSIX and Windows

    @staticmethod
    def _append(path: Path, record: Mapping[str, object]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _read_lines(path: Path) -> Iterator[dict[str, object]]:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # a torn final line from an interrupted append: ignore (recovery, not corruption)
                    continue
