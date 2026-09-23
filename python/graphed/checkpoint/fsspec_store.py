"""The checkpoint store at a URL: :class:`FsspecStore`, on any fsspec filesystem.

The layout mirrors :class:`~graphed.checkpoint.Store`, with one change forced by object stores,
which cannot append in place: each log file becomes a prefix of one-record objects.

- ``<root>/objects/<sha256>`` holds the blobs, with the same names and bytes as the local store.
- ``<root>/journal.log/``, ``<root>/journal.<node>.log/`` and ``<root>/dead_letter.log/`` hold one
  object per record, each the exact line ``Store`` would append for the same call. The object name
  is ``<writer>-<seq>``: ``writer`` is fixed per instance (creation time plus a random id) and
  ``seq`` counts that instance's records. Records replay in name order, so order is exact within
  an instance and follows creation time across instances.

A blob write is one whole-object write, and ``get`` verifies the bytes against the name. A torn or
tampered object is therefore never served; the next ``put`` of the true bytes rewrites it.
Identical-content writers need no exclusion, because they write the same bytes to the same name.
"""

from __future__ import annotations

import glob
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .store import JournalEntry, Store, _done_record, _parse_record, _record_line, _replay


class FsspecStore:
    """A :class:`~graphed.checkpoint.CheckpointStore` rooted at an fsspec ``url``.

    ``storage_options`` pass through to :func:`fsspec.core.url_to_fs` (credentials, endpoint, ...).
    ``node`` selects the writer's own journal prefix, as for :class:`~graphed.checkpoint.Store`.
    A root belongs to one store class: ``Store`` and ``FsspecStore`` cannot share a directory,
    because a log is a file for one and a prefix for the other.
    """

    def __init__(self, url: str, node: str | None = None, **storage_options: Any) -> None:
        try:
            from fsspec.core import url_to_fs  # noqa: PLC0415  (lazy: fsspec is the optional extra)
        except ImportError as exc:
            raise ImportError(
                "FsspecStore needs fsspec — install the optional extra: pip install 'graphed[checkpoint]'"
            ) from exc
        self.url = url
        self.node = node
        self.fs, self.root = url_to_fs(url, **storage_options)
        self.objects = f"{self.root}/objects"
        journal_name = "journal.log" if node is None else f"journal.{node}.log"
        self.journal_path = f"{self.root}/{journal_name}"
        self.dead_letter_path = f"{self.root}/dead_letter.log"
        # file:// does not create parent directories on write
        for prefix in (self.objects, self.journal_path, self.dead_letter_path):
            self.fs.makedirs(prefix, exist_ok=True)
        self._writer = f"{time.time_ns():020d}-{uuid.uuid4().hex}"
        self._seq = 0
        self._lock = threading.Lock()

    # ---- content-addressed blobs ----------------------------------------------------------------
    def put(self, data: bytes) -> str:
        """Store ``data`` under its content hash and return the hash; a copy that does not verify
        is rewritten."""
        digest = Store.content_hash(data)
        if self.get(digest) is None:
            self.fs.pipe_file(f"{self.objects}/{digest}", data)
        return digest

    def get(self, digest: str) -> bytes | None:
        """The blob named ``digest``, or ``None`` when it is absent or its bytes do not hash to it."""
        try:
            data: bytes = self.fs.cat_file(f"{self.objects}/{digest}")
        except FileNotFoundError:
            return None
        return data if Store.content_hash(data) == digest else None

    # ---- manifest / journal ---------------------------------------------------------------------
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
        """Replay the union of every writer's journal prefix into ``task_id -> JournalEntry``."""
        present = {path.rsplit("/", 1)[-1] for path in self.fs.find(self.objects)}
        return _replay(self._records(f"{glob.escape(self.root)}/journal*.log/*"), present.__contains__)

    # ---- dead-letter set ------------------------------------------------------------------------
    def record_dead(self, descriptor: Mapping[str, object]) -> None:
        self._append(self.dead_letter_path, dict(descriptor))

    def dead_letters(self) -> list[dict[str, object]]:
        return self._records(f"{glob.escape(self.dead_letter_path)}/*")

    # ---- internals ------------------------------------------------------------------------------
    def _append(self, prefix: str, record: Mapping[str, object]) -> None:
        # a lock, not itertools.count: free-threaded builds make no promise for the latter
        with self._lock:
            seq = self._seq
            self._seq += 1
        self.fs.pipe_file(f"{prefix}/{self._writer}-{seq:012d}", _record_line(record).encode())

    def _records(self, pattern: str) -> list[Any]:
        # the root is escaped because a user's path is data, not glob syntax
        paths = sorted(self.fs.glob(pattern))
        if not paths:
            return []
        # async fsspec (2025.9 and older) returns a failed read in the list whatever on_error says
        data = self.fs.cat_ranges(paths, [None] * len(paths), [None] * len(paths))
        for raw in data:
            if isinstance(raw, Exception):
                raise raw
        return [rec for rec in map(_parse_record, data) if rec is not None]
