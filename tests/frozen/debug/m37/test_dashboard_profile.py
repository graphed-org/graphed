"""M37 frozen suite (graphed-debug slice): the profile path on the Dashboard itself — merging
per-worker sampler sessions into one flamegraph, the worker-profiler factory gate, malformed-payload
robustness, and ``attach``/``monitor`` wiring."""

from __future__ import annotations

from graphed_debug import Dashboard, _sampler


def _profiled_payload(spins: int) -> bytes:
    p = _sampler.make_worker_profiler()
    p.start()
    s = 0.0
    for i in range(spins):
        s += (i % 5) ** 0.5
    payload = p.stop()
    assert payload is not None
    return payload


def test_on_profile_merges_workers_into_flame() -> None:
    d = Dashboard(profile=True)
    d.on_profile("w0", _profiled_payload(500000))
    snap = d.snapshot()
    assert snap["flame"]["children"]  # one worker's samples populate the tree
    d.on_profile("w1", _profiled_payload(500000))
    assert len(d._sessions) == 2  # a second worker merges in
    assert d.snapshot()["flame"]["value"] > 0


def test_on_profile_ignores_malformed_payload() -> None:
    d = Dashboard(profile=True)
    d.on_profile("w0", b"not a session")  # must not raise
    assert d.snapshot()["flame"] == {"name": "root", "value": 0, "children": []}


def test_worker_profiler_factory_gated_by_profile_flag() -> None:
    assert Dashboard(profile=True).worker_profiler_factory() is _sampler.make_worker_profiler
    assert Dashboard(profile=False).worker_profiler_factory() is None


def test_attach_sets_monitor_and_returns_executor() -> None:
    class FakeExecutor:
        monitor: object = None

        def run(self, plan: object) -> object:
            return None

    d = Dashboard()
    fake = FakeExecutor()
    assert d.attach(fake) is fake
    assert fake.monitor is d
    assert d.monitor is d
