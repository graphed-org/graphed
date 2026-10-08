"""m74 T4: plans resume refuses, before any store I/O, and plans it accepts (plan §2.5, §4.1 T4)."""

from __future__ import annotations

import functools
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import m74_helpers as h
import pytest

from graphed import Session, aggregate_plan, collate, repartition, shuffle_plan
from graphed.awkward import AwkwardBackend, gak
from graphed.awkward.io import from_parquet, to_parquet
from graphed.checkpoint import check_resumable, resumable
from graphed.core import DurablePlanV2, Partition, Plan
from graphed.services import ServiceSpec

SPEC = ServiceSpec("sf", "http")


def _bare(p: Any, r: Any) -> float:
    return float(p.entry_start)


def _with_services(process: Any, uri: str = "mem://m74/events") -> Plan[Any]:
    plan = h.plan_of(process, h.partitions(uri=uri))
    return Plan(process=plan.process, combine=plan.combine, empty=plan.empty, tasks=plan.tasks, services=(SPEC,))


def refused(plan: Any, root: Path, *needles: str) -> None:
    with pytest.raises(TypeError) as checked:
        check_resumable(plan)
    with pytest.raises(TypeError) as made:
        resumable(plan, str(root))
    assert not root.exists()
    for needle in needles:
        assert needle in str(checked.value)
        assert needle in str(made.value)


def test_plan_with_next_tasks(tmp_path: Path) -> None:
    plan = Plan(process=h.marked, combine=h.add, empty=h.zero, next_tasks=lambda ctx: None)
    refused(plan, tmp_path / "store", "next_tasks")


def test_services_with_an_undeclared_bare_function(tmp_path: Path) -> None:
    refused(_with_services(_bare), tmp_path / "store", "checkpointable")


def test_services_with_a_served_wrapper_is_refused_unpickled(tmp_path: Path) -> None:
    h.PICKLED.clear()
    refused(_with_services(h.Served(h.marked)), tmp_path / "store", "checkpointable")
    assert h.PICKLED == []


def test_collated_plan_holding_a_served_wrapper(tmp_path: Path) -> None:
    h.PICKLED.clear()
    plan = collate({"served": _with_services(h.Served(h.marked)), "plain": _with_services(h.marked, "mem://m74/other")})
    refused(plan, tmp_path / "store", "checkpointable")
    assert h.PICKLED == []


def test_process_holding_a_lock(tmp_path: Path) -> None:
    lock = threading.Lock()

    def locked(p: Any, r: Any) -> float:
        with lock:
            return float(p.entry_start)

    refused(h.plan_of(locked), tmp_path / "store", "lock")


def test_process_reading_a_main_cache_def(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @functools.cache
    def scale(x: float) -> float:
        return x * 2.0

    scale.__module__ = "__main__"
    scale.__qualname__ = "m74_scale"
    monkeypatch.setattr(sys.modules["__main__"], "m74_scale", scale, raising=False)
    refused(h.plan_of(lambda p, r: scale(float(p.entry_start))), tmp_path / "store", "__main__.m74_scale")


def test_already_resumable_plan(tmp_path: Path) -> None:
    rp = resumable(h.plan_of(h.marked), str(tmp_path / "first"))
    refused(rp, tmp_path / "second")


@pytest.fixture(scope="module")
def paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    pytest.importorskip("pyarrow")
    return h.events(tmp_path_factory.mktemp("m74refusals"))


def _keyed(plan: Any) -> list[Partition]:
    if isinstance(plan, DurablePlanV2):
        return [t.partition for stage in plan.stages for t in stage.tasks]
    return [t.partition for t in plan.tasks]


def _lambda_plan(paths: list[str]) -> Any:
    return h.plan_of(lambda p, r: float(p.entry_start) * 2.0)


def _aggregate_with_services(paths: list[str]) -> Any:
    return aggregate_plan(gak.sum(h.service_events(paths)), reduce=h.agg_reduce, combine=h.agg_combine, empty=h.agg_empty)


def _shuffle(paths: list[str]) -> Any:
    ev = from_parquet(Session(AwkwardBackend()), "events", paths, open_files=False, steps_per_file=2)
    return shuffle_plan(gak.sum(repartition(ev, n=2).x), reduce=h.agg_reduce, combine=h.agg_combine, empty=h.agg_empty)


def _write_with_services(paths: list[str]) -> Any:
    return to_parquet(h.service_events(paths), str(Path(paths[0]).parent / "written"), compute=False)


@pytest.mark.parametrize(
    "build", [_lambda_plan, _aggregate_with_services, _shuffle, _write_with_services], ids=lambda f: f.__name__
)
def test_allowed(tmp_path: Path, paths: list[str], build: Callable[[list[str]], Any]) -> None:
    plan = build(paths)
    if build in (_aggregate_with_services, _write_with_services):
        assert plan.services
    assert check_resumable(plan) is None
    rp = resumable(plan, str(tmp_path / "store"))
    keyed = _keyed(rp)
    assert keyed
    assert all(type(p) is not Partition and p.task_id for p in keyed)
