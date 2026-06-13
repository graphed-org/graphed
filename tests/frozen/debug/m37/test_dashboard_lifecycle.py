"""M37 frozen suite (graphed-debug slice): the dashboard is **opt-in** — no server/thread/port until
``start()``/``__enter__``, and a clean teardown on ``stop()``/``__exit__`` (the anti-greedy contrast
with dask's always-on server)."""

from __future__ import annotations

import urllib.error
import urllib.request

from graphed_debug import Dashboard


def test_constructing_starts_nothing() -> None:
    d = Dashboard()
    assert d._server is None
    assert d._thread is None  # no background thread until start()


def test_start_then_stop_lifecycle() -> None:
    d = Dashboard().start()
    thread = d._thread
    assert d._server is not None and thread is not None and thread.is_alive()
    base = d.url.rstrip("/")
    with urllib.request.urlopen(base + "/api/state.json", timeout=5) as resp:
        assert resp.status == 200

    d.stop()
    assert d._server is None
    assert not thread.is_alive()  # the serve thread joined
    # the port is no longer served
    try:
        urllib.request.urlopen(base + "/api/state.json", timeout=2)
        raise AssertionError("server still answering after stop()")
    except (urllib.error.URLError, OSError):
        pass


def test_start_is_idempotent() -> None:
    d = Dashboard().start()
    try:
        first = d._server
        assert d.start()._server is first  # second start() is a no-op
    finally:
        d.stop()


def test_context_manager_starts_and_stops() -> None:
    with Dashboard() as d:
        thread = d._thread
        assert thread is not None and thread.is_alive()
    assert not thread.is_alive()
