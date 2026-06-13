"""M37 frozen suite (graphed-debug slice): an errored task surfaces in the dashboard's error panel
with the user's analysis line — the same source-mapped ``StageError`` rendering as M6 (the executor
ships ``str(StageError)`` as the event's error summary; here we feed that exact string)."""

from __future__ import annotations

from graphed_core import TaskEvent, TaskPhase

from graphed_debug import Dashboard, StageError
from graphed_debug.errors import SourceFrame


def _stage_error() -> StageError:
    return StageError(
        op="divide",
        frames=(SourceFrame(filename="analysis.py", lineno=42, function="select", source="z = a / b"),),
        input_forms=("float64", "float64"),
        partition="f.root:Events:0-100",
        cause_type="ZeroDivisionError",
        cause_message="division by zero",
        opt_level=1,
    )


def test_errored_task_surfaces_user_line_in_panel() -> None:
    err = _stage_error()
    summary = str(err)  # what graphed-exec-local's _render_error ships for a StageError
    assert "analysis.py:42" in summary and "divide" in summary  # the M6 rendering carries the user line

    d = Dashboard()
    d.on_task(TaskEvent(TaskPhase.SUBMITTED, 3, "driver", 0.0, "f.root:Events:0-100", 100))
    d.on_task(TaskEvent(TaskPhase.STARTED, 3, "w1", 0.1, "f.root:Events:0-100", 100))
    d.on_task(TaskEvent(TaskPhase.ERRORED, 3, "w1", 0.2, "f.root:Events:0-100", 100, error=summary))

    snap = d.snapshot()
    assert snap["counters"]["errored"] == 1
    assert snap["inflight"] == 0  # the errored task left flight
    panel = snap["last_error"]
    assert panel is not None
    assert panel["key"] == 3 and panel["worker"] == "w1"
    assert "analysis.py:42" in panel["message"] and "divide" in panel["message"]
