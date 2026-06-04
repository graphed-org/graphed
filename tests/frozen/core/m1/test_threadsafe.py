"""Concurrency: many threads building overlapping subgraphs yield exactly the distinct-key count
with no panic/race (M1). Runs under the GIL and, in CI, under free-threaded 3.14t."""

from __future__ import annotations

import sysconfig
import threading

import graphed_core as gc


def test_concurrent_overlapping_builds_intern_consistently() -> None:
    store = gc.GraphStore()
    src = store.add_source("events", {"uri": "f.root"})

    n_threads = 16
    n_ops = 50  # each thread builds the SAME 50 ops -> heavy interning contention
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        try:
            barrier.wait()
            for i in range(n_ops):
                store.add_op(f"op{i}", [src], {"i": i})
        except BaseException as exc:  # noqa: BLE001 - record any race/panic
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"
    # all threads built the same nodes -> exactly source + n_ops distinct nodes
    assert store.node_count() == 1 + n_ops


def test_concurrent_distinct_builds_count_exactly() -> None:
    store = gc.GraphStore()
    src = store.add_source("events")
    n_threads = 8
    per = 100
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(per):
                store.add_op("k", [src], {"t": tid, "i": i})  # disjoint across threads
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"
    assert store.node_count() == 1 + n_threads * per


def test_reports_whether_free_threaded() -> None:
    # informational: the same test body runs under 3.14t in CI where this is True
    gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    assert gil_disabled in (0, 1, None)
