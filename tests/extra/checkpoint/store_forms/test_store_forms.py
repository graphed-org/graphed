"""`run_resumable` and `resumable` take a path, an `os.PathLike` or a store alike."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from graphed.checkpoint import FsspecStore, Store, resumable, run_resumable
from graphed.checkpoint import resume as resume_module
from graphed.checkpoint.resume import as_store
from graphed.checkpoint.runner import run_shuffle_resumable
from graphed.core import DurablePlan, OpSpec, Partition, SequentialRunner


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


def test_a_form_written_by_one_resumes_under_another(tmp_path: Path) -> None:
    run_resumable(_plan(), str(tmp_path))
    assert run_resumable(_plan(), Store(tmp_path)).report.skipped == 3


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


def test_a_store_passes_through_untouched(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert as_store(store) is store


class _NoEnvironment:
    pass


@pytest.mark.parametrize("value", [3, None, _NoEnvironment()])
def test_run_resumable_names_store_on_anything_else(value: Any) -> None:
    with pytest.raises(TypeError, match="is a path, a URL or a Store, got"):
        run_resumable(_plan(), value)


def test_run_shuffle_resumable_takes_the_same_forms() -> None:
    with pytest.raises(TypeError, match="is a path, a URL or a Store, got int"):
        run_shuffle_resumable(_plan(), 3)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [3, None])
def test_resumable_names_store_on_anything_else(value: Any) -> None:
    with pytest.raises(TypeError, match="takes a path, a URL or a Store, got"):
        resumable(_plan(), value)


def test_resumable_sends_an_fsspec_store_to_its_url(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="goes as its URL with storage_options="):
        resumable(_plan(), FsspecStore(f"file://{tmp_path}"))  # type: ignore[call-overload]
