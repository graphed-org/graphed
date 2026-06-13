"""M37 frozen suite (graphed-debug slice): the web server — static routes, the JSON snapshot, the
SSE stream (snapshot then live deltas), and the path-traversal guard."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from graphed_core import TaskEvent, TaskPhase

from graphed_debug import Dashboard


def _get(url: str) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def test_static_and_api_routes() -> None:
    with Dashboard() as d:
        base = d.url.rstrip("/")
        status, body, ctype = _get(base + "/")
        assert status == 200 and b"graphed" in body and "text/html" in ctype
        status, body, ctype = _get(base + "/static/uPlot.min.js")
        assert status == 200 and len(body) > 1000
        status, body, ctype = _get(base + "/static/app.js")
        assert status == 200 and b"EventSource" in body
        status, body, ctype = _get(base + "/api/state.json")
        assert status == 200 and "application/json" in ctype
        assert json.loads(body)["type"] == "snapshot"


def test_unknown_route_is_404() -> None:
    with Dashboard() as d:
        try:
            _get(d.url.rstrip("/") + "/does-not-exist")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


def test_path_traversal_is_blocked() -> None:
    # http.client sends the raw path WITHOUT client-side normalization, so the server's resolve()
    # guard is actually exercised.
    with Dashboard() as d:
        parsed = urlparse(d.url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("GET", "/static/../../pyproject.toml")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        assert resp.status == 404  # escaping the static dir is refused


def test_sse_streams_snapshot_then_delta() -> None:
    with Dashboard() as d:
        resp = urllib.request.urlopen(d.url.rstrip("/") + "/events", timeout=5)
        assert resp.headers.get("Content-Type") == "text/event-stream"

        def next_frame() -> dict[str, object]:
            while True:
                line = resp.readline()
                if not line:
                    raise AssertionError("stream closed before a frame arrived")
                if line.startswith(b"data: "):
                    return json.loads(line[len(b"data: ") :])

        snap = next_frame()
        assert snap["type"] == "snapshot"
        # a live event must produce a delta frame: a STARTED bumps in-flight
        d.on_task(TaskEvent(TaskPhase.STARTED, 0, "w0", 0.1, "f.root:Events:0-10", 10))
        delta = next_frame()
        assert delta["type"] == "stat"
        assert delta["inflight"] == 1
        resp.close()
