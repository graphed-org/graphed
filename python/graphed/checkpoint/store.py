"""The content-addressed checkpoint Store (plan M8).

A local-filesystem store with three durable parts (:mod:`graphed.checkpoint.fsspec_store` keeps the
same parts at a URL):

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
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class JournalEntry:
    """One completed task recorded in the manifest. M39 adds ``stage`` (which pipeline stage the
    block belongs to — ``map_write``/``gather_join``/``manifest``/…) and ``deps`` (the upstream input
    block hashes for a gather block), so a multi-stage shuffle resumes with the right dependency
    structure. Both default empty, so a V1 single-stage entry is unchanged."""

    task_id: str
    partition: str  # a human-readable partition tag (uri@start:stop), for audit
    blob: str  # content hash of the stored output
    stage: str = ""
    deps: tuple[str, ...] = ()


def _record_line(record: Mapping[str, object]) -> str:
    """The one serialization of a journal or dead-letter record, shared by every store."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


def _parse_record(raw: str | bytes) -> Any:
    """A parsed record, or ``None`` for a torn one (an interrupted append, or bad UTF-8)."""
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _done_record(
    task_id: str, partition: str, blob: str, stage: str, deps: tuple[str, ...]
) -> dict[str, object]:
    # ``stage``/``deps`` only when set, so a plain single-stage record stays byte-identical to the
    # V1 journal line (the M8 determinism gate)
    rec: dict[str, object] = {"task_id": task_id, "partition": partition, "blob": blob}
    if stage:
        rec["stage"] = stage
    if deps:
        rec["deps"] = list(deps)
    return rec


def _replay(records: Iterable[Any], present: Callable[[str], bool]) -> dict[str, JournalEntry]:
    """``task_id -> JournalEntry`` from records in write order (the later record wins), honouring
    only a record whose blob is present (a journal line can outrace its object write across a
    crash)."""
    done: dict[str, JournalEntry] = {}
    for rec in records:
        blob = rec.get("blob")
        if isinstance(blob, str) and present(blob):
            tid = str(rec.get("task_id", ""))
            raw_deps = rec.get("deps", [])
            deps = tuple(str(d) for d in raw_deps) if isinstance(raw_deps, list) else ()
            done[tid] = JournalEntry(
                tid, str(rec.get("partition", "")), blob, str(rec.get("stage", "")), deps
            )
    return done


@runtime_checkable
class CheckpointStore(Protocol):
    """The store the resumable runners take: content-addressed blobs, a replayable journal of
    completed tasks, and a dead-letter set.

    Contract beyond the signatures: ``get`` returns ``None`` for bytes that do not hash to their
    name (a corrupted result is recomputed, never served), and concurrent ``put`` of identical bytes
    never fails."""

    def put(self, data: bytes) -> str: ...

    def get(self, digest: str) -> bytes | None: ...

    def record_done(
        self,
        task_id: str,
        partition: str,
        blob: str,
        *,
        stage: str = "",
        deps: tuple[str, ...] = (),
    ) -> None: ...

    def completed(self) -> dict[str, JournalEntry]: ...

    def record_dead(self, descriptor: Mapping[str, object]) -> None: ...

    def dead_letters(self) -> list[dict[str, object]]: ...


class Store:
    """A content-addressed, append-only, crash-safe checkpoint store on the local filesystem.

    M39 (§6.3/§7.3): with ``node=None`` this is the V1 single-writer store — one ``journal.log``,
    byte-for-byte the M8 behaviour. With ``node="A"`` it writes its OWN ``journal.A.log`` (so N nodes
    checkpointing shuffle intermediates to one root never contend on a shared append), and
    :meth:`completed` replays the UNION of every writer's journal."""

    def __init__(self, root: str | os.PathLike[str], node: str | None = None) -> None:
        self.root = Path(root)
        self.node = node
        self.objects = self.root / "objects"
        journal_name = "journal.log" if node is None else f"journal.{node}.log"
        self.journal_path = self.root / journal_name
        self.dead_letter_path = self.root / "dead_letter.log"
        self.objects.mkdir(parents=True, exist_ok=True)

    # ---- content-addressed blobs ----------------------------------------------------------------
    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def put(self, data: bytes) -> str:
        """Store ``data`` under its content hash, atomically and idempotently. Returns the hash.

        A present object whose bytes do not verify is rewritten, so a put heals a corrupted blob."""
        digest = self.content_hash(data)
        if self.get(digest) is None:
            self._atomic_write(digest, data)
        return digest

    def has_blob(self, digest: str) -> bool:
        return (self.objects / digest).exists()

    def get(self, digest: str) -> bytes | None:
        """The blob named ``digest``, or ``None`` when it is absent or its bytes do not hash to it."""
        try:
            data = (self.objects / digest).read_bytes()
        except FileNotFoundError:
            return None
        return data if self.content_hash(data) == digest else None

    # ---- append-only manifest / journal ---------------------------------------------------------
    def record_done(
        self,
        task_id: str,
        partition: str,
        blob: str,
        *,
        stage: str = "",
        deps: tuple[str, ...] = (),
    ) -> None:
        self._append(self.journal_path, _done_record(task_id, partition, blob, stage, deps))

    def completed(self) -> dict[str, JournalEntry]:
        """Replay the UNION of every writer's journal (``journal.log`` + ``journal.<node>.log``) into
        ``task_id -> JournalEntry`` (last write wins, deterministic file order). A torn trailing line
        (interrupted append) is skipped, never fatal."""
        journals = sorted(self.root.glob("journal*.log"))
        return _replay((rec for path in journals for rec in self._read_lines(path)), self.has_blob)

    # ---- dead-letter set ------------------------------------------------------------------------
    def record_dead(self, descriptor: Mapping[str, object]) -> None:
        self._append(self.dead_letter_path, dict(descriptor))

    def dead_letters(self) -> list[dict[str, object]]:
        return list(self._read_lines(self.dead_letter_path))

    # ---- internals ------------------------------------------------------------------------------
    def _atomic_write(self, digest: str, data: bytes) -> None:
        # a per-call temp in the SAME directory (rename stays atomic); builtin open keeps the
        # default file mode, where mkstemp would make every blob 0600
        dest = self.objects / digest
        tmp = dest.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "xb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp, dest)
            except OSError:
                # Windows refuses to replace a file another writer holds open; that writer's copy
                # is as good as ours once it verifies
                if self.get(digest) is None:
                    raise
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _append(path: Path, record: Mapping[str, object]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_record_line(record))
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _read_lines(path: Path) -> Iterator[dict[str, object]]:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = _parse_record(line)
                if rec is not None:
                    yield rec
