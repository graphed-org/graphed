"""M37 frozen suite (graphed-debug slice): the ``Dashboard`` as a passive ``Monitor`` — counters,
in-flight tracking, per-worker rollup, and bounded buffers, driven by direct event feeds (no
executor dependency, per R13.8)."""

from __future__ import annotations

from graphed_core import TaskEvent, TaskPhase

from graphed_debug import Dashboard


def _ev(phase: TaskPhase, key: int, worker: str = "w0", t: float = 0.0, n: int = 10) -> TaskEvent:
    return TaskEvent(phase, key, worker, t, f"f{key}.root:Events:0-{n}", n)


def test_counters_inflight_and_worker_rollup() -> None:
    d = Dashboard()
    for k in range(3):
        d.on_task(_ev(TaskPhase.SUBMITTED, k))
    for k in range(3):
        d.on_task(_ev(TaskPhase.STARTED, k, worker="wA"))
    d.on_task(_ev(TaskPhase.FINISHED, 0, worker="wA"))
    snap = d.snapshot()
    assert snap["counters"]["submitted"] == 3
    assert snap["counters"]["started"] == 3
    assert snap["counters"]["finished"] == 1
    assert snap["inflight"] == 2
    assert snap["workers"]["wA"]["started"] == 3
    assert snap["workers"]["wA"]["finished"] == 1
    assert snap["workers"]["wA"]["entries"] == 10


def test_on_combine_counts() -> None:
    d = Dashboard()
    d.on_combine(1)
    d.on_combine(2)
    assert d.snapshot()["counters"]["combines"] == 2


def test_throughput_ring_is_bounded() -> None:
    d = Dashboard(ring=8)
    for k in range(50):
        d.on_task(_ev(TaskPhase.FINISHED, k))
    snap = d.snapshot()
    assert len(snap["throughput"]) <= 8  # bounded by the ring parameter
    assert len(snap["recent"]) <= 256  # fixed recent-event ring


def test_snapshot_shape_before_any_event() -> None:
    snap = Dashboard().snapshot()
    assert snap["type"] == "snapshot"
    assert snap["counters"] == {"submitted": 0, "started": 0, "finished": 0, "errored": 0, "combines": 0}
    assert snap["inflight"] == 0
    assert snap["flame"] == {"name": "root", "value": 0, "children": []}
    assert snap["last_error"] is None
