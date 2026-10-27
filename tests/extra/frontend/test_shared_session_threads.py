"""Two threads registering variations on one Session: the `vary` transaction is the unit."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import graphed
from graphed import GraphedError, Session
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record


class _PausingBackend(NumpyBackend):
    """Holds the thread named `pause_in` inside the form check `vary` runs after it minted."""

    def __init__(self) -> None:
        super().__init__()
        self.pause_in: threading.Thread | None = None
        self.inside = threading.Event()
        self.resume = threading.Event()

    def op_form(self, op: str, inputs: Any, params: Any) -> Any:
        if op == "add" and threading.current_thread() is self.pause_in:
            self.inside.set()
            self.resume.wait(timeout=0.5)
        return super().op_form(op, inputs, params)


def _guarded(fn: Callable[[], None], errors: list[BaseException]) -> threading.Thread:
    def run() -> None:
        try:
            fn()
        except BaseException as exc:  # a thread's failure, pytest's included, is the caller's
            errors.append(exc)

    return threading.Thread(target=run)


def _run(*targets: Callable[[], None]) -> list[BaseException]:
    errors: list[BaseException] = []
    threads = [_guarded(fn, errors) for fn in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_a_failing_vary_cannot_erase_another_threads_mints() -> None:
    backend = _PausingBackend()
    session = Session(backend)
    record = from_record(session, "ev", x=np.arange(3.0), k=np.arange(3))
    x, k = record["x"], record["k"]
    other = from_record(session, "other", y=np.arange(3.0))["y"]

    def failing() -> None:
        # `a_up`'s int form pauses this thread in the form check; `a_down` is then refused
        with pytest.raises(GraphedError, match="different source"):
            graphed.vary(x, "a", up=k, down=other)

    def minting() -> None:
        backend.inside.wait()
        graphed.vary(x, "b", up=x * 2.0)
        backend.resume.set()

    errors: list[BaseException] = []
    failing_thread, minter = _guarded(failing, errors), _guarded(minting, errors)
    backend.pause_in = failing_thread
    failing_thread.start()
    minter.start()
    failing_thread.join()
    minter.join()
    assert errors == []
    assert backend.inside.is_set(), "the failing registration never reached the paused check"
    assert "b_up" in session._points, "a rollback erased another thread's mint"
    assert "a_up" not in session._points


@pytest.fixture
def _fast_switching() -> Iterator[None]:
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


@pytest.mark.usefixtures("_fast_switching")
def test_explain_during_a_concurrent_vary_does_not_raise() -> None:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", x=np.arange(3.0), w=np.ones(3))
    x = record["x"]
    root = EventContext(session, x, collections={"x": x, "w": record["w"]})
    ctx = graphed.vary(root, "sf", record["w"], is_weight=True, up=record["w"] * 1.1)
    for index in range(200):
        graphed.vary(x, f"seed{index}", up=x * (1.0 + index))
    done = threading.Event()

    def minting() -> None:
        try:
            for index in range(300):
                graphed.vary(x, f"f{index}", up=x * (2.0 + index))
        finally:
            done.set()

    def explaining() -> None:
        while not done.is_set():
            graphed.explain(ctx)

    assert _run(minting, explaining) == []
