"""M37 frozen suite (graphed-debug slice): the pyinstrument sampler shim + flamegraph adapter.

Uses the real ``dashboard`` extra (pyinstrument is in dev deps). Validates: a worker profiler
produces a serialized session, sessions combine across workers, and the adapter yields a valid
d3-flame-graph ``{name, value, children}`` tree whose parent values cover their children."""

from __future__ import annotations

from typing import Any

from graphed_debug import _sampler


def _spin(n: int) -> float:
    s = 0.0
    for i in range(n):
        s += (i % 5) ** 0.5
    return s


def _check_tree(node: dict[str, Any]) -> None:
    assert set(node) == {"name", "value", "children"}
    child_sum = sum(c["value"] for c in node["children"])
    assert node["value"] >= child_sum or not node["children"]
    for child in node["children"]:
        _check_tree(child)


def test_profiler_session_serializes_and_combines() -> None:
    assert _sampler.pyinstrument_available()
    p = _sampler.make_worker_profiler()
    p.start()
    _spin(400000)
    b1 = p.flush()
    _spin(400000)
    b2 = p.stop()
    assert b1 and b2  # heavy work -> both intervals captured samples
    combined = _sampler.combine_sessions(_sampler.session_from_bytes(b1), _sampler.session_from_bytes(b2))
    tree = _sampler.flamegraph_tree(combined)
    _check_tree(tree)
    assert tree["children"]  # the spin loop shows up
    assert tree["value"] > 0


def test_empty_profiler_flush_is_none() -> None:
    p = _sampler.make_worker_profiler()
    # never started -> nothing to flush/stop
    assert p.flush() is None
    assert p.stop() is None


class _FakeFrame:
    def __init__(self, function: str, time: float, children: tuple = (), pos: str | None = None) -> None:
        self.function = function
        self.time = time
        self.children = list(children)
        self._pos = pos

    def code_position_short(self) -> str | None:
        return self._pos


class _BadPosFrame(_FakeFrame):
    def code_position_short(self) -> str | None:
        raise RuntimeError("boom")


def test_frame_to_tree_label_both_branches() -> None:
    leaf = _FakeFrame("inner", 0.01, pos="m.py:5")
    root = _FakeFrame("outer", 0.03, (leaf,), pos=None)  # no position -> bare name
    tree = _sampler._frame_to_tree(root)
    assert tree["name"] == "outer"
    assert tree["children"][0]["name"] == "inner  (m.py:5)"
    assert tree["value"] >= tree["children"][0]["value"]


def test_frame_label_survives_a_raising_position() -> None:
    assert _sampler._frame_label(_BadPosFrame("f", 0.0)) == "f"


def test_flamegraph_tree_handles_no_samples() -> None:
    class _EmptySession:
        def root_frame(self) -> None:
            return None

    assert _sampler.flamegraph_tree(_EmptySession()) == {"name": "(no samples)", "value": 0, "children": []}
