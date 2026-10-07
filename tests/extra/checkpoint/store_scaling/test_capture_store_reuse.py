"""``aggregate_plan(store=<url>)`` opens the capture store once per worker thread and passes ``storage_options`` to it."""

from __future__ import annotations

import collections
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import awkward as ak
import numpy as np

import graphed.checkpoint as gcp
from graphed import Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.core import SequentialRunner

OPTIONS = {"default_cache_type": "readahead"}


def _first(v: list[Any]) -> float:
    return float(v[0])


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def _plan(tmp_path: Path, url: str, **kw: Any) -> Any:
    path = tmp_path / "d.parquet"
    ak.to_parquet(ak.Array({"x": np.arange(1000.0)}), path)
    y = gak.sum(from_parquet(Session(AwkwardBackend()), "events", str(path)).x, axis=None)
    return aggregate_plan(y, reduce=_first, combine=_add, empty=_zero, steps_per_file=4, store=url, **kw)


def test_capture_store_is_opened_once_and_gets_storage_options(
    tmp_path: Path, s3_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = gcp.FsspecStore(s3_url, **OPTIONS)  # the filesystem instance (and client) the tasks share
    calls: collections.Counter[str] = collections.Counter()
    client = probe.fs.s3
    client.meta.events.register("before-call.s3", lambda model, **_k: calls.update([model.name]))
    gcp.FsspecStore(s3_url, "one-open", **OPTIONS)
    one_open = calls["HeadBucket"]
    calls.clear()

    opened: list[dict[str, Any]] = []
    init = gcp.FsspecStore.__init__

    def spy(self: Any, url: str, node: str | None = None, **opts: Any) -> None:
        opened.append(opts)
        init(self, url, node, **opts)

    monkeypatch.setattr(gcp.FsspecStore, "__init__", spy)
    plan = _plan(tmp_path, s3_url, storage_options=OPTIONS)
    assert len(plan.tasks) == 4
    SequentialRunner().run(plan)

    assert opened == [OPTIONS]
    assert calls["HeadBucket"] <= one_open
    assert len(gcp.FsspecStore(s3_url, **OPTIONS).completed()) == 8


def test_local_root_ignores_storage_options(tmp_path: Path) -> None:
    plan = _plan(tmp_path, str(tmp_path / "cap"), storage_options=OPTIONS)
    SequentialRunner().run(plan)
    assert len(gcp.Store(str(tmp_path / "cap")).completed()) == 8


def test_a_removed_local_root_is_reopened_by_a_rerun(tmp_path: Path) -> None:
    cap = tmp_path / "ck"
    SequentialRunner().run(_plan(tmp_path, str(cap)))
    shutil.rmtree(cap)
    SequentialRunner().run(_plan(tmp_path, str(cap)))
    assert len(gcp.Store(cap).completed()) == 8
