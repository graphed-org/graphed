"""m62 unit B — ``FsspecStore`` meets the ``CheckpointStore`` contract at a URL."""

from __future__ import annotations

import hashlib
import os
import random
import stat
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyses
import m62_url_helpers as h
import numpy as np
import pytest

from graphed.checkpoint import CheckpointStore, FsspecStore, Store, run_resumable


def test_fsspec_store_satisfies_the_protocol() -> None:
    url = f"memory://m62-{uuid.uuid4().hex}"
    store = FsspecStore(url)
    assert isinstance(store, CheckpointStore)
    url_to_fs = pytest.importorskip("fsspec.core").url_to_fs
    assert store.url == url
    assert store.node is None
    assert store.root == url_to_fs(url)[1]
    assert store.objects == f"{store.root}/objects"
    assert store.journal_path == f"{store.root}/journal.log"
    assert store.dead_letter_path == f"{store.root}/dead_letter.log"
    assert FsspecStore(url, node="A").journal_path == f"{store.root}/journal.A.log"
    assert FsspecStore(url).fs is store.fs
    assert FsspecStore(url, skip_instance_cache=True).fs is not store.fs


def test_url_get_refuses_bytes_that_do_not_hash_to_their_name(store_url) -> None:
    store = FsspecStore(store_url)
    digest = store.put(b"true bytes")
    store.fs.pipe_file(f"{store.objects}/{digest}", b"evil bytes")
    assert store.get(digest) is None
    assert store.put(b"true bytes") == digest
    assert store.get(digest) == b"true bytes"
    assert store.fs.cat_file(f"{store.objects}/{digest}") == b"true bytes"


def test_url_resume_recomputes_a_corrupted_partial_and_heals_it(store_url) -> None:
    plan = h.hist_plan(6)
    assert run_resumable(plan, FsspecStore(store_url)).report.executed == 6
    blobs = [e.blob for e in FsspecStore(store_url).completed().values()]
    victim = next(b for b in blobs if blobs.count(b) == 1)
    store = FsspecStore(store_url)
    store.fs.pipe_file(f"{store.objects}/{victim}", b"corrupted partial")

    healed = run_resumable(plan, FsspecStore(store_url))
    assert healed.report.executed == 1
    assert healed.report.skipped == 5
    assert np.array_equal(healed.value, analyses.reference())
    assert hashlib.sha256(store.fs.cat_file(f"{store.objects}/{victim}")).hexdigest() == victim

    again = run_resumable(plan, FsspecStore(store_url))
    assert again.report.executed == 0
    assert np.array_equal(again.value, analyses.reference())


def test_url_concurrent_identical_puts_never_fail(store_url) -> None:
    for trial in range(20):
        store = FsspecStore(f"{store_url}/t{trial}")
        data = random.Random(trial).randbytes(1 << 20)
        barrier = threading.Barrier(8)

        def put(_: int, store: FsspecStore = store, data: bytes = data, barrier: threading.Barrier = barrier) -> str:
            barrier.wait()
            return store.put(data)

        with ThreadPoolExecutor(8) as pool:
            digests = list(pool.map(put, range(8)))
        digest = Store.content_hash(data)
        assert digests == [digest] * 8
        assert store.get(digest) == data
        assert [p.rsplit("/", 1)[-1] for p in store.fs.find(store.objects)] == [digest]


def test_url_put_keeps_the_default_file_mode(tmp_path) -> None:
    old = os.umask(0o022)
    try:
        store = FsspecStore("file://" + (tmp_path / "store").as_posix())
        digest = store.put(b"mode witness")
        sibling = tmp_path / "sibling"
        with open(sibling, "wb") as f:
            f.write(b"mode witness")
    finally:
        os.umask(old)
    blob = Path(f"{store.objects}/{digest}")
    assert blob.read_bytes() == b"mode witness"
    assert stat.S_IMODE(blob.stat().st_mode) == stat.S_IMODE(sibling.stat().st_mode)


def _records_survive_glob_metacharacters(root_a, root_b) -> None:
    a, b = root_a(), root_b()
    blob_a = a.put(b"partial-A")
    a.record_done("task-A", "uri@0:10", blob_a)
    blob_b = b.put(b"partial-B")
    b.record_done("task-B", "uri@10:20", blob_b)
    a.record_dead({"task_id": "t1", "error_type": "ValueError"})
    b.record_dead({"task_id": "t2", "error_type": "MemoryError"})
    reader = root_a()
    done = reader.completed()
    assert set(done) == {"task-A", "task-B"}
    assert done["task-A"].blob == blob_a
    assert done["task-B"].blob == blob_b
    assert sorted(d["task_id"] for d in reader.dead_letters()) == ["t1", "t2"]


def test_root_with_glob_metacharacters_keeps_its_records(store_url) -> None:
    root = f"{store_url}/run[x]"
    _records_survive_glob_metacharacters(lambda: FsspecStore(root), lambda: FsspecStore(root, node="B"))


def test_local_root_with_glob_metacharacters_keeps_its_records(tmp_path) -> None:
    root = tmp_path / "run[x]"
    _records_survive_glob_metacharacters(lambda: Store(root), lambda: Store(root, node="B"))
