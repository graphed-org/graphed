"""``Store.put`` of a present blob must not read it back."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphed.checkpoint import Store


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    real = Path.read_bytes
    sizes: list[int] = []

    def read_bytes(self: Path) -> bytes:
        data = real(self)
        sizes.append(len(data))
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return sizes


def test_put_of_a_present_blob_reads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(tmp_path)
    data = b"x" * 4096
    digest = store.put(data)
    sizes = _count_reads(monkeypatch)
    assert store.put(data) == digest
    assert sizes == []


def test_put_rewrites_a_blob_that_get_refused(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = b"y" * 64
    digest = store.put(data)
    (store.objects / digest).write_bytes(b"z" * 64)
    assert store.get(digest) is None
    assert store.put(data) == digest
    assert store.get(digest) == data
