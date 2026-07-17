"""M37 browser smoke test (graphed-debug slice): load the real dashboard page in headless Chromium
and assert it renders without JS errors. This is the regression guard for the class of bug that the
Python-only tests cannot see — a Perspective client/server version mismatch or an unregistered plugin
shows up only as browser-side ``cannot read properties of null`` / red error overlays.

It is heavy (a real browser) and gated three ways so it never burdens the normal matrix:
``importorskip`` Playwright + Perspective, and it skips unless a browser is actually installed. CI
runs it in a dedicated job (``playwright install chromium``). Locally::

    pip install playwright && playwright install chromium
    pytest tests/frozen/m37/test_dashboard_browser.py
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("playwright")
pytest.importorskip("perspective")

from playwright.sync_api import sync_playwright

from graphed.core import TaskEvent, TaskPhase
from graphed.debug import Dashboard, _sampler


def _ev(phase: TaskPhase, key: int, worker: str = "w0") -> TaskEvent:
    return TaskEvent(phase, key, worker, float(key), f"f{key}.root:Events:0-{key}", key)


def _profile_payload(duration: float = 0.2) -> bytes:
    prof = _sampler.make_worker_profiler()
    prof.start()
    end = time.monotonic() + duration
    s = 0.0
    i = 0
    while time.monotonic() < end:
        s += (i % 5) ** 0.5
        i += 1
    payload = prof.stop()
    assert payload is not None
    return payload


def test_dashboard_page_renders_in_a_browser() -> None:
    with Dashboard(profile=True) as dash:
        for k in range(4):
            dash.monitor.on_task(_ev(TaskPhase.SUBMITTED, k))
            dash.monitor.on_task(_ev(TaskPhase.STARTED, k))
            dash.monitor.on_task(_ev(TaskPhase.FINISHED, k))
        dash.monitor.on_combine(3)
        dash.monitor.on_profile("w0", _profile_payload())
        dash.wait_for(finished=4, timeout=20)

        console_errors: list[str] = []
        page_errors: list[str] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch()  # the CI job runs `playwright install chromium` first
            page = browser.new_page()
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(dash.url, wait_until="load")
            # stats is a Datagrid -> renders a <regular-table> once it has loaded its table from the
            # websocket. If the client/server versions mismatched, or a plugin were missing, it never
            # appears and/or the error lists fill.
            page.wait_for_selector("#stats regular-table", timeout=30000, state="attached")
            # the progress panel polls /api/progress.json and renders one .pbar-row per bar (overall +
            # per worker) -> at least the "all tasks" bar appears.
            page.wait_for_selector("#progress .pbar-row", timeout=30000, state="attached")
            # the profile panel is a d3 flamegraph polling /api/flamegraph.json -> once samples have
            # been ingested it draws an <svg class="d3-flame-graph"> into #profile.
            page.wait_for_selector("#profile svg.d3-flame-graph", timeout=30000, state="attached")

            # hover a flamegraph cell -> the shared pop-out tooltip shows the FULL name + sample count
            # (the cells truncate the name; this is the whole point of the mouseover).
            page.hover("#profile svg.d3-flame-graph g")
            page.wait_for_function(
                "document.getElementById('gp-tooltip').style.display === 'block'", timeout=5000
            )
            assert "samples" in page.inner_text("#gp-tooltip")
            # hover an individual per-worker TASK cell -> the tooltip shows that one task's details
            page.hover("#progress .pbar-cells .tcell")
            page.wait_for_function(
                "document.getElementById('gp-tooltip').innerText.includes('task')", timeout=5000
            )

            time.sleep(0.5)  # let any late errors surface
            browser.close()

        assert console_errors == [], f"browser console errors: {console_errors}"
        assert page_errors == [], f"browser page errors: {page_errors}"
