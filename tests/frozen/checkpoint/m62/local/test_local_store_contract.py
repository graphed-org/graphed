"""m62 unit A — the local ``Store`` meets the ``CheckpointStore`` contract."""

from __future__ import annotations

import hashlib
import os
import random
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyses
import numpy as np
import pytest

from graphed.checkpoint import CheckpointStore, Store, run_resumable

_SIX = ("completed", "get", "put", "record_done", "record_dead", "dead_letters")


def _method(self, *args, **kwargs):
    return None


@pytest.mark.parametrize("omitted", _SIX)
def test_store_satisfies_the_protocol(tmp_path, omitted) -> None:
    assert isinstance(Store(tmp_path), CheckpointStore)
    five = type("Five", (), {name: _method for name in _SIX if name != omitted})
    six = type("Six", (), dict.fromkeys(_SIX, _method))
    assert not isinstance(five(), CheckpointStore)
    assert isinstance(six(), CheckpointStore)


def test_get_refuses_bytes_that_do_not_hash_to_their_name(tmp_path) -> None:
    s = Store(tmp_path)
    h = s.put(b"true bytes")
    (s.objects / h).write_bytes(b"evil bytes")
    assert s.get(h) is None
    assert s.put(b"true bytes") == h
    assert s.get(h) == b"true bytes"
    assert (s.objects / h).read_bytes() == b"true bytes"


def test_resume_recomputes_a_corrupted_partial_and_heals_it(tmp_path) -> None:
    plan = analyses.build_plan("analyses:histogram_chunk", 6)
    first = run_resumable(plan, Store(tmp_path))
    assert first.report.executed == 6
    entries = list(Store(tmp_path).completed().values())
    blobs = [e.blob for e in entries]
    victim = next(b for b in blobs if blobs.count(b) == 1)
    (Store(tmp_path).objects / victim).write_bytes(b"corrupted partial")

    healed = run_resumable(plan, Store(tmp_path))
    assert healed.report.executed == 1
    assert healed.report.skipped == 5
    assert np.array_equal(healed.value, analyses.reference())
    assert hashlib.sha256((Store(tmp_path).objects / victim).read_bytes()).hexdigest() == victim

    again = run_resumable(plan, Store(tmp_path))
    assert again.report.executed == 0
    assert np.array_equal(again.value, analyses.reference())


def test_concurrent_identical_puts_never_fail(tmp_path) -> None:
    for trial in range(20):
        s = Store(tmp_path / f"t{trial}")
        data = random.Random(trial).randbytes(1 << 20)
        barrier = threading.Barrier(8)

        def put(_: int, s: Store = s, data: bytes = data, barrier: threading.Barrier = barrier) -> str:
            barrier.wait()
            return s.put(data)

        with ThreadPoolExecutor(8) as pool:
            digests = list(pool.map(put, range(8)))
        h = Store.content_hash(data)
        assert digests == [h] * 8
        assert s.get(h) == data
        assert sorted(p.name for p in s.objects.iterdir()) == [h]


@pytest.mark.parametrize("copy_present", [True, False], ids=["present-copy", "no-copy"])
def test_put_survives_losing_the_rename_race_to_a_present_copy(tmp_path, monkeypatch, copy_present) -> None:
    data = b"the committed partial"
    h = Store.content_hash(data)
    s = Store(tmp_path)
    calls: list[str] = []
    lost = PermissionError(13, "sharing violation")

    def fake(src, dst, *args, **kwargs):
        calls.append(Path(dst).name)
        if copy_present:
            Path(dst).write_bytes(data)
        raise lost

    monkeypatch.setattr(os, "replace", fake)
    if copy_present:
        assert s.put(data) == h
        monkeypatch.undo()
        assert s.get(h) == data
        assert sorted(p.name for p in s.objects.iterdir()) == [h]
    else:
        with pytest.raises(PermissionError) as raised:
            s.put(data)
        assert raised.value is lost
        monkeypatch.undo()
        assert s.get(h) is None
        assert sorted(p.name for p in s.objects.iterdir()) == []
    assert calls
    assert set(calls) == {h}


def test_put_keeps_the_default_file_mode(tmp_path) -> None:
    old = os.umask(0o022)
    try:
        s = Store(tmp_path / "store")
        h = s.put(b"mode witness")
        sibling = tmp_path / "sibling"
        with open(sibling, "wb") as f:
            f.write(b"mode witness")
    finally:
        os.umask(old)
    assert stat.S_IMODE((s.objects / h).stat().st_mode) == stat.S_IMODE(sibling.stat().st_mode)
