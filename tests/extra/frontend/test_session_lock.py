"""One Session shared by threads: its methods lock its state; evaluation never holds the lock."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from graphed import Session, apply, compile_ir
from graphed.aggregate import external_evaluators
from graphed.numpy import NumpyBackend, from_array


@pytest.fixture(autouse=True)
def _fast_switching() -> Iterator[None]:
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


def _race(
    background: Callable[[threading.Event], None], foreground: Callable[[], None]
) -> list[str]:
    """Run ``foreground`` against ``background`` threads until it returns; return their errors."""
    stop = threading.Event()
    errors: list[str] = []

    def guarded(fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                fn()
            except Exception as exc:  # a thread's failure is the test's
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                stop.set()

        return run

    threads = [threading.Thread(target=guarded(lambda: background(stop))) for _ in range(4)]
    threads.append(threading.Thread(target=guarded(foreground)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_compiling_while_other_threads_record_raises_nothing() -> None:
    session = Session(NumpyBackend())
    x = from_array(session, "x", np.arange(10.0))
    out = apply(lambda a: a + 1, x, output_type="float64").sum()
    counter = iter(range(10**9))

    def record(stop: threading.Event) -> None:
        while not stop.is_set():
            i = next(counter)
            own = from_array(session, f"src{i}", np.arange(3.0))
            apply(lambda a, i=i: a * i, own, output_type="float64")

    def compile_repeatedly() -> None:
        for _ in range(150):
            external_evaluators(session, compile_ir(session, out))
            session.sourcemap()

    assert _race(record, compile_repeatedly) == []


def _finishes(fn: Callable[[], object], timeout: float = 5.0) -> bool:
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive()


def test_the_lock_is_reentrant() -> None:
    session = Session(NumpyBackend())
    x = from_array(session, "x", np.arange(3.0))

    def nested() -> None:
        with session._lock:
            _ = x + 1.0
            _ = session.systematics.points

    assert _finishes(nested)


def test_evaluation_does_not_hold_the_lock() -> None:
    session = Session(NumpyBackend())
    x = from_array(session, "x", np.arange(3.0))
    entered, release = threading.Event(), threading.Event()

    def blocking(a: Any) -> Any:
        entered.set()
        release.wait(5.0)
        return a + 1

    out = apply(blocking, x, output_type="float64")
    values: list[object] = []
    evaluating = threading.Thread(target=lambda: values.append(session.materialize(out)))
    evaluating.start()
    try:
        assert entered.wait(5.0)
        assert _finishes(lambda: from_array(session, "y", np.arange(2.0)) + 1.0, timeout=2.0)
    finally:
        release.set()
        evaluating.join()
    np.testing.assert_array_equal(values[0], np.arange(3.0) + 1)
