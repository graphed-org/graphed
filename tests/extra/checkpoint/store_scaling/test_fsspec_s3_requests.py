"""Requests ``FsspecStore.completed`` issues against S3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from graphed.checkpoint import FsspecStore

_N = 12


def test_completed_reads_each_journal_record_once(
    s3_url: str, s3_counter: Callable[..., Counter[str]]
) -> None:
    writer = FsspecStore(s3_url)
    for i in range(_N):
        writer.record_done(f"t{i}", "p", writer.put(f"b{i}".encode()))
    reader = FsspecStore(s3_url)
    done: dict[str, object] = {}
    calls = s3_counter(reader, lambda: done.update(reader.completed()))
    assert len(done) == _N
    assert calls["GetObject"] == _N
    if hasattr(reader.fs, "max_concurrency"):  # older s3fs sizes every read with its own HEAD
        assert calls["HeadObject"] <= 2, "HEADs must not scale with the record count"


def test_a_large_blob_is_still_read_in_concurrent_ranges(
    s3_url: str, s3_counter: Callable[..., Counter[str]]
) -> None:
    store = FsspecStore(s3_url)
    if not hasattr(store.fs, "max_concurrency"):  # older s3fs has no concurrent ranged reads
        return
    big = bytes(range(256)) * (204 * 1024)  # 51 MiB, past s3fs's 50 MiB default block size
    digest = store.put(big)
    calls = s3_counter(store, lambda: store.get(digest))
    assert calls["GetObject"] > 1
