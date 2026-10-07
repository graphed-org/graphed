"""Requests ``FsspecStore.put`` issues against S3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from graphed.checkpoint import FsspecStore


def test_put_of_an_existing_blob_neither_reads_nor_writes_it(
    s3_url: str, s3_counter: Callable[..., Counter[str]]
) -> None:
    store = FsspecStore(s3_url)
    data = b"x" * 4096
    store.put(data)
    calls = s3_counter(store, lambda: store.put(data))
    assert calls["GetObject"] == 0
    assert calls["PutObject"] == 0
    assert calls["HeadObject"] == 1


def test_put_of_a_new_blob_writes_it(s3_url: str, s3_counter: Callable[..., Counter[str]]) -> None:
    store = FsspecStore(s3_url)
    calls = s3_counter(store, lambda: store.put(b"fresh"))
    assert calls["PutObject"] == 1


def test_put_rewrites_a_blob_that_get_refused(s3_url: str) -> None:
    store = FsspecStore(s3_url)
    data = b"y" * 64
    digest = store.put(data)
    store.fs.pipe_file(f"{store.objects}/{digest}", b"z" * 64)
    assert store.get(digest) is None
    assert store.put(data) == digest
    assert store.get(digest) == data
