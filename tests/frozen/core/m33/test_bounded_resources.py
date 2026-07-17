"""M33 — LocalResources is bounded and closeable (no unbounded handle accumulation).

`open_once` is a within-run/per-worker locality cache; on a long-lived worker (a persistent
process pool over many files) it MUST NOT accumulate every handle for the worker's lifetime.
The set of simultaneously-open handles is bounded (LRU-evicted, closing the evicted handle), and
`close()` releases all of them; the SequentialRunner closes its resources at end of run.
"""

from __future__ import annotations

from graphed.core import LocalResources, SequentialRunner
from graphed.core.execution import Partition, Plan, Task


class _Handle:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_open_once_reuses_within_the_bound() -> None:
    res = LocalResources(max_open=8)
    opens: list[str] = []
    h1 = res.open_once("a", lambda u: opens.append(u) or _Handle(u))
    h2 = res.open_once("a", lambda u: opens.append(u) or _Handle(u))
    assert h1 is h2  # reused, not reopened
    assert opens == ["a"]
    assert res.open_count == 1


def test_exceeding_the_bound_closes_the_least_recently_used() -> None:
    res = LocalResources(max_open=2)
    handles = {u: res.open_once(u, _Handle) for u in ("a", "b")}
    res.open_once("a", _Handle)  # touch 'a' -> 'b' is now the LRU
    res.open_once("c", _Handle)  # over the bound -> evict + close 'b'
    assert handles["b"].closed is True
    assert handles["a"].closed is False  # still open (was most-recently-used)
    assert res.open_count == 3


def test_reopen_after_eviction(monkeypatch=None) -> None:  # type: ignore[no-untyped-def]
    res = LocalResources(max_open=1)
    opens: list[str] = []

    def opener(uri: str) -> _Handle:
        opens.append(uri)
        return _Handle(uri)

    res.open_once("a", opener)
    res.open_once("b", opener)  # evicts 'a'
    res.open_once("a", opener)  # 'a' was evicted -> reopened
    assert opens == ["a", "b", "a"]
    assert res.open_count == 3


def test_close_releases_every_handle() -> None:
    res = LocalResources()
    hs = [res.open_once(u, _Handle) for u in ("a", "b", "c")]
    res.close()
    assert all(h.closed for h in hs)  # type: ignore[attr-defined]
    # reusable after close: a fresh open works
    again = res.open_once("a", _Handle)
    assert again.closed is False


def test_sequential_runner_closes_resources_after_the_run() -> None:
    captured: list[_Handle] = []

    def proc(partition: Partition, resources: object) -> list[int]:
        h = resources.open_once(partition.uri, _Handle)  # type: ignore[attr-defined]
        captured.append(h)
        return [1]

    tasks = tuple(Task(i, Partition(f"f{i}", "", 0, 1)) for i in range(3))
    SequentialRunner().run(Plan(process=proc, combine=lambda a, b: a + b, empty=list, tasks=tasks))
    assert len(captured) == 3
    assert all(h.closed for h in captured)  # the runner closed its LocalResources in finally
