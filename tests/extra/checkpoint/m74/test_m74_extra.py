"""m74 paths the frozen suite reaches only in subprocesses, or not at all: the environment record in
process, the key rules for ABC registries, ``__main__`` modules and untracked reductions, the worker
store cache, and the wrappers' forwarding hooks."""

from __future__ import annotations

import abc
import os
import pickle
import sys
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from graphed.checkpoint import EnvironmentChanged, Store, check_resumable, resumable
from graphed.checkpoint import resume as rs
from graphed.core import (
    DurablePlanV2,
    LocalResources,
    OpSpec,
    Partition,
    Plan,
    SequentialRunner,
    StageSpec,
    Task,
)
from graphed.services import bind_services, resolve_services

T = 4


def _parts() -> tuple[Partition, ...]:
    return tuple(Partition("mem://m74x", "Events", 10 * i, 10 * i + 10) for i in range(T))


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _plan(process: Any) -> Plan[Any]:
    return Plan(
        process=process, combine=_add, empty=_zero, tasks=tuple(Task(i, p) for i, p in enumerate(_parts()))
    )


def _reused(process: Any, root: Path) -> int:
    rp: Any = resumable(_plan(process), str(root))
    count: int = rp.process.reused
    return count


def _filled(process: Any, root: Path) -> None:
    SequentialRunner().run(resumable(_plan(process), str(root)))


def _dist(root: Path, name: str, version: str, *, named: bool = True) -> Path:
    info = root / f"{name}-{version}" / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    header = f"Name: {name}\n" if named else ""
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\n{header}Version: {version}\n", encoding="utf-8")
    return info.parent


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_key_bytes_are_never_loaded() -> None:
    def process(p: Any, r: Any) -> float:
        return 1.0

    with pytest.raises(AssertionError, match="never loaded"):
        pickle.loads(rs._key_bytes(process))


def test_the_main_module_is_refused(tmp_path: Path) -> None:
    main = sys.modules["__main__"]

    def process(p: Any, r: Any) -> float:
        return float(hasattr(main, "x"))

    with pytest.raises(TypeError, match="__main__ module"):
        check_resumable(_plan(process))


def _registry(order: Sequence[str]) -> Any:
    class Scaled(abc.ABC):
        @abc.abstractmethod
        def scale(self) -> int: ...

    class Registry(Scaled):
        def scale(self) -> int:
            return 1

        def __call__(self, p: Any, r: Any) -> float:
            return float(p.entry_start + self.scale())

    made = {n: type(n, (), {"f": lambda self: 1}) for n in order}
    for n in sorted(made):
        Registry.register(made[n])
    return Registry()


def test_an_abc_registry_made_in_another_order_is_reused(tmp_path: Path) -> None:
    names = [f"C{i}" for i in range(12)]
    _filled(_registry(names), tmp_path)
    assert _reused(_registry(names[::-1]), tmp_path) == T


V = typing.TypeVar("V")


def test_reductions_without_a_tracker_id_are_kept(tmp_path: Path) -> None:
    def make(kind: Any) -> Any:
        def process(p: Any, r: Any) -> float:
            return float(p.entry_start) if kind in (type(None), V) else -1.0

        return process

    _filled(make(type(None)), tmp_path)
    assert _reused(make(type(None)), tmp_path) == T
    assert _reused(make(V), tmp_path) == 0


def test_the_environment_record_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, dists = tmp_path / "store", tmp_path / "dists"
    _filled(lambda p, r: 2.0, root)
    monkeypatch.syspath_prepend(str(_dist(dists, "zznameless", "1.0", named=False)))
    assert _reused(lambda p, r: 2.0, root) == T
    before = _files(root)
    monkeypatch.syspath_prepend(str(_dist(dists, "zzfake", "1.0")))
    monkeypatch.setattr(sys.implementation, "cache_tag", "cpython-00")
    with pytest.raises(EnvironmentChanged) as info:
        resumable(_plan(lambda p, r: 2.0), str(root))
    assert "zzfake 1.0 added" in str(info.value)
    assert "-> cpython-00" in str(info.value)
    assert _files(root) == before
    accepted: Any = resumable(_plan(lambda p, r: 2.0), str(root), accept_environment=True)
    assert accepted.process.reused == T
    assert len(Store(root).completed().environments) == 2


def test_behaviors_are_empty_without_awkward(monkeypatch: pytest.MonkeyPatch) -> None:
    import awkward as ak  # noqa: PLC0415

    monkeypatch.setitem(ak.behavior, "m74x", _add)
    assert rs._behaviors() == (("m74x", _add),)
    monkeypatch.delitem(sys.modules, "awkward")
    assert rs._behaviors() == ()


def test_worker_stores_per_process_and_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "memory://m74x-cache"
    assert rs.task_store(url) is rs.task_store(url)
    directory = str(tmp_path / "dir")
    assert rs.task_store(directory) is not rs.task_store(directory)
    monkeypatch.setattr(rs, "_TOKEN", (os.getpid() + 1, "inherited"))
    node = rs.task_store(str(tmp_path / "forked")).node
    assert not node.startswith("inherited")
    assert node.startswith(rs._TOKEN[1])


class _Writer:
    """A process with a ``part_paths`` hook."""

    def __call__(self, p: Partition, r: Any) -> float:
        return float(p.entry_start)

    def part_paths(self, p: Partition) -> list[str]:
        return [f"out/{type(p).__name__}-{p.entry_start}"]


def test_part_paths_reach_the_inner_process(tmp_path: Path) -> None:
    rp: Any = resumable(_plan(_Writer()), str(tmp_path / "a"))
    task = rp.tasks[1]
    assert type(task.partition) is not Partition
    assert list(rp.process.part_paths(task.partition)) == ["out/Partition-10"]
    bare: Any = resumable(_plan(lambda p, r: 1.0), str(tmp_path / "b"))
    assert list(bare.process.part_paths(task.partition)) == []


class _Stage:
    """A stage process with service hooks: it tags its payload with the bound endpoint."""

    def __init__(self, endpoint: str = "") -> None:
        self.endpoint = endpoint

    def __call__(self, task: Task, inputs: Sequence[bytes], resources: Any) -> bytes:
        return f"{self.endpoint}:{task.key}:{len(inputs)}".encode()

    def bind_services(self, endpoints: Mapping[str, str]) -> _Stage:
        return _Stage(endpoints["svc"])

    def resolve_services(self, value: Any) -> Any:
        return ("resolved", value)


def _staged(process: Any) -> DurablePlanV2:
    tasks = tuple(Task(i, Partition("src", "e", i, i + 1)) for i in range(2))
    return DurablePlanV2(
        ir=b"m74x",
        stages=(StageSpec("map_write", process=OpSpec("ref", "m74x:stage", live=process), tasks=tasks),),
    )


def _stage_fn(task: Task, inputs: Sequence[bytes], resources: Any) -> bytes:
    return b"x"


def test_stage_wrapper_forwards_service_hooks(tmp_path: Path) -> None:
    rp = resumable(_staged(_Stage()), str(tmp_path / "s"))
    bound = bind_services(rp, {"svc": "tcp://host:1"})
    process = bound.stages[0].process.resolve()
    assert process is not rp.stages[0].process.resolve()
    assert process(bound.stages[0].tasks[0], (), LocalResources()) == b"tcp://host:1:0:0"
    assert resolve_services(bound, "v") == ("resolved", "v")
    bare = resumable(_staged(_stage_fn), str(tmp_path / "t"))
    unbound: Any = bind_services(bare, {"svc": "tcp://host:1"}).stages[0].process.resolve()
    assert unbound.inner is _stage_fn
    assert resolve_services(bare, "v") == "v"


BEHAVIOR: dict[str, Any] = {"m74x": _add}


def test_a_writer_behavior_reference_resolves_for_the_key() -> None:
    from graphed.awkward.io import _resolve_behavior  # noqa: PLC0415

    assert _resolve_behavior("test_m74_extra:BEHAVIOR") is BEHAVIOR
    assert _resolve_behavior(BEHAVIOR) is BEHAVIOR
