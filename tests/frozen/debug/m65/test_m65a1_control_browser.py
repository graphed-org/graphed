"""M65 A1 browser test (graphed-debug slice): the dashboard page's control bar drives a
``RunControl`` in headless Chromium. Runs in the ``dashboard-smoke`` CI job."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("playwright")
pytest.importorskip("perspective")

from m65a1_control_helpers import until
from playwright.sync_api import sync_playwright

import graphed.core as gc
from graphed.debug import Dashboard


def _state_text_contains(page: Any, word: str) -> None:
    page.wait_for_function(
        f"(document.querySelector('#ctl-state')?.innerText || '').toLowerCase().includes('{word}')",
        timeout=10000,
    )


def _is_progress_poll(response: Any) -> bool:
    return "api/progress.json" in response.url


def test_control_buttons_drive_the_run_control() -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            with Dashboard(control=True) as dash:
                until(lambda: dash.server.snapshot()["control_listeners"] == 1)
                page = browser.new_page()
                page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                page.goto(dash.url, wait_until="load")
                page.wait_for_selector("#ctl-pause", state="visible", timeout=30000)
                for button, state, word in (
                    ("#ctl-pause", gc.RunState.PAUSED, "paused"),
                    ("#ctl-resume", gc.RunState.RUNNING, "running"),
                    ("#ctl-cancel", gc.RunState.CANCELLED, "cancelled"),
                ):
                    page.click(button)
                    until(lambda state=state: dash.control.state is state, timeout=10)
                    _state_text_contains(page, word)
                page.close()

            with Dashboard() as dash:
                page = browser.new_page()
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                with page.expect_response(_is_progress_poll, timeout=30000):
                    page.goto(dash.url, wait_until="load")
                page.wait_for_event("response", _is_progress_poll, timeout=30000)
                assert not page.is_visible("#ctl")
                page.close()
        finally:
            browser.close()

    assert console_errors == [], f"browser console errors: {console_errors}"
    assert page_errors == [], f"browser page errors: {page_errors}"
