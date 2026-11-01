"""`run_resumable` and `resumable` take a path, an `os.PathLike` or a store alike."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from graphed.checkpoint import FsspecStore, Store, resumable, run_resumable
from graphed.checkpoint import resume as resume_module
from graphed.checkpoint.runner import run_shuffle_resumable
from graphed.core import (
    DurablePlan,
    DurablePlanV2,
    OpSpec,
    Partition,
    SequentialRunner,
    StageSpec,
    Task,
)
from graphed.services import ServiceSpec, UnboundServiceError


def process(p: Partition, _r: Any) -> int:
    return p.entry_stop - p.entry_start


def combine(a: int, b: int) -> int:
    return a + b


def empty() -> int:
    return 0


def _plan() -> DurablePlan:
    def op(fn: Any) -> OpSpec:
        return OpSpec("ref", f"t:{fn.__name__}", live=fn)

    return DurablePlan(
        ir=b"store-forms",
        process=op(process),
        combine=op(combine),
        empty=op(empty),
        partitions=tuple(Partition(f"mem://{i}", "t", 0, i + 1) for i in range(3)),
    )


def write(task: Task, inputs: Sequence[bytes], resources: Any) -> bytes:
    return f"block:{task.key}".encode()


class _NeedsService:
    def bind_services(self, endpoints: Mapping[str, str]) -> _NeedsService:
        if "sf" not in endpoints:
            raise UnboundServiceError("sf")
        return self


def _shuffle(process: Any = write, **fields: Any) -> DurablePlanV2:
    tasks = tuple(Task(i, Partition("src", "e", i, i + 1)) for i in range(2))
    stage = StageSpec("map_write", process=OpSpec("ref", "t:write", live=process), tasks=tasks)
    return DurablePlanV2(ir=b"store-forms", stages=(stage,), **fields)


FORMS: list[Callable[[Path], Any]] = [str, Path, Store]


@pytest.mark.parametrize("form", FORMS)
def test_run_resumable_runs_then_resumes_from_each_form(
    form: Callable[[Path], Any], tmp_path: Path
) -> None:
    first = run_resumable(_plan(), form(tmp_path))
    assert (first.value, first.report.executed, first.report.skipped) == (6, 3, 0)
    again = run_resumable(_plan(), form(tmp_path))
    assert (again.value, again.report.executed, again.report.skipped) == (6, 0, 3)
    assert (tmp_path / "journal.log").exists()


@pytest.mark.parametrize("form", FORMS)
def test_resumable_runs_then_reuses_from_each_form(
    form: Callable[[Path], Any], tmp_path: Path
) -> None:
    assert SequentialRunner().run(resumable(_plan(), form(tmp_path))).value == 6
    plan: Any = resumable(_plan(), form(tmp_path))
    assert plan.process.reused == 3


def test_a_url_takes_the_route_resumable_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []
    real = resume_module.open_store

    def spying(root: str, node: str | None = None, options: Any = None) -> Any:
        calls.append((root, node, options))
        return real(str(tmp_path), node)

    monkeypatch.setattr(resume_module, "open_store", spying)
    run_resumable(_plan(), "s3://bucket/ckpt", storage_options={"anon": True})
    assert calls == [("s3://bucket/ckpt", None, {"anon": True})]


@pytest.mark.parametrize(
    ("given", "options"),
    [({}, {"auto_mkdir": True}), ({"extra": 1}, {"auto_mkdir": True, "extra": 1})],
)
def test_resumable_sends_workers_an_fsspec_stores_url_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: dict[str, Any], options: dict[str, Any]
) -> None:
    url = f"file://{tmp_path}/fsspec"
    store = FsspecStore(url, auto_mkdir=True)
    calls: list[tuple[Any, ...]] = []
    real = resume_module.open_store

    def spying(root: str, node: str | None = None, opened: Any = None) -> Any:
        calls.append((root, opened))
        return real(root, node, {"auto_mkdir": True})

    monkeypatch.setattr(resume_module, "open_store", spying)
    plan: Any = resumable(_plan(), store, storage_options=given)
    assert calls == [(url, options)]
    assert plan.process.storage_options == tuple(options.items())


def test_an_fsspec_store_runs_then_reuses_under_resumable(tmp_path: Path) -> None:
    store = FsspecStore(f"file://{tmp_path}/fsspec", auto_mkdir=True)
    assert SequentialRunner().run(resumable(_plan(), store)).value == 6
    plan: Any = resumable(_plan(), store)
    assert plan.process.reused == 3


@pytest.mark.parametrize(
    "call",
    [
        lambda value: run_resumable(_plan(), value),
        lambda value: run_shuffle_resumable(_shuffle(), value),
        lambda value: resumable(_plan(), value),
    ],
)
@pytest.mark.parametrize("value", [3, None])
def test_each_entry_point_names_store_on_anything_else(
    call: Callable[[Any], Any], value: Any
) -> None:
    with pytest.raises(TypeError, match="a path, a URL or a Store, got"):
        call(value)


@pytest.mark.parametrize("form", FORMS)
def test_run_shuffle_resumable_runs_then_resumes_from_each_form(
    form: Callable[[Path], Any], tmp_path: Path
) -> None:
    first = run_shuffle_resumable(_shuffle(), form(tmp_path))
    assert (first.report.executed, first.report.skipped) == (2, 0)
    again = run_shuffle_resumable(_shuffle(), form(tmp_path))
    assert (again.report.executed, again.report.skipped) == (0, 2)
    assert again.value == first.value


def test_run_resumable_refuses_a_plan_before_opening_the_store(tmp_path: Path) -> None:
    root = tmp_path / "store"
    plan = replace(_plan(), process=OpSpec("ref", "no.such.module:process"))
    with pytest.raises(ModuleNotFoundError):
        run_resumable(plan, root)
    assert not root.exists()


def test_run_shuffle_resumable_refuses_a_plan_before_opening_the_store(tmp_path: Path) -> None:
    root = tmp_path / "store"
    plan = _shuffle(_NeedsService(), services=(ServiceSpec("sf", "http"),))
    with pytest.raises(UnboundServiceError):
        run_shuffle_resumable(plan, root)
    assert not root.exists()


def test_resumable_refuses_the_store_before_keying_the_plan(tmp_path: Path) -> None:
    already: Any = resumable(_plan(), tmp_path)
    with pytest.raises(TypeError, match="takes a path, a URL or a Store, got int"):
        resumable(already, 3)  # type: ignore[call-overload]
