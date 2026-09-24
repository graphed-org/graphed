"""A blob that a concurrent put is replacing is briefly unopenable on Windows; ``get`` waits it out."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from graphed.checkpoint import Store
from graphed.checkpoint import store as store_module


def _flaky_read_bytes(monkeypatch: pytest.MonkeyPatch, failures: int) -> list[int]:
    real = Path.read_bytes
    seen = [0]

    def read_bytes(self: Path) -> bytes:
        seen[0] += 1
        if seen[0] <= failures:
            raise PermissionError(13, "delete pending")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return seen


def test_get_waits_out_a_transient_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Store(tmp_path)
    h = s.put(b"mid-replace")
    seen = _flaky_read_bytes(monkeypatch, failures=2)
    assert s.get(h) == b"mid-replace"
    assert seen[0] == 3


def test_get_gives_up_after_the_backoff_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Store(tmp_path)
    h = s.put(b"stuck")
    seen = _flaky_read_bytes(monkeypatch, failures=len(store_module._READ_BACKOFF) + 1)
    with pytest.raises(PermissionError):
        s.get(h)
    assert seen[0] == len(store_module._READ_BACKOFF) + 1


def test_a_missing_blob_is_still_none_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Store(tmp_path)
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    assert s.get("0" * 64) is None
    assert slept == []
