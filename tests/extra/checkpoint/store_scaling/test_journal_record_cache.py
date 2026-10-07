"""A journal record is read once per process however many ``completed()`` listings see it."""

from __future__ import annotations

import collections
import fnmatch
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("pyarrow")

import awkward as ak
import numpy as np

import graphed.checkpoint as gcp
import graphed.debug as gd
from graphed import Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, from_parquet, gak
from graphed.core import SequentialRunner


def _first(v: list[Any]) -> float:
    return float(v[0])


def _add(a: float, b: float) -> float:
    return a + b


def _zero() -> float:
    return 0.0


def test_replays_of_one_capture_root_read_each_record_once(
    tmp_path: Path, s3_url: str, s3_counter: Callable[..., collections.Counter[str]]
) -> None:
    path = tmp_path / "d.parquet"
    ak.to_parquet(ak.Array({"x": np.arange(1000.0)}), path)
    y = gak.sum(from_parquet(Session(AwkwardBackend()), "events", str(path)).x, axis=None)
    plan = aggregate_plan(y, reduce=_first, combine=_add, empty=_zero, steps_per_file=4, store=s3_url)
    SequentialRunner().run(plan)
    probe = gcp.FsspecStore(s3_url)

    first = s3_counter(probe, lambda: gd.replay(plan, 0, y))
    again = s3_counter(probe, lambda: [gd.replay(plan, k, y) for k in range(4)])

    assert first["GetObject"] == 8
    assert again["GetObject"] == 0
    assert gd.replay(plan, 3, y).input_source == "store"


def test_records_written_after_a_listing_appear_in_the_next(tmp_path: Path) -> None:
    url = (tmp_path / "root").as_uri()
    a, b = gcp.FsspecStore(url, "a"), gcp.FsspecStore(url, "b")
    a.record_done("t1", "p1", a.put(b"one"))
    assert set(a.completed()) == {"t1"}
    b.record_done("t2", "p2", b.put(b"two"))
    assert set(a.completed()) == {"t1", "t2"}


def test_dead_letters_are_not_shared_through_the_cache(tmp_path: Path) -> None:
    store = gcp.FsspecStore((tmp_path / "root").as_uri())
    store.record_dead({"task_id": "t"})
    store.dead_letters()[0]["task_id"] = "mutated"
    assert store.dead_letters()[0]["task_id"] == "t"


class _ListProtocolFS:
    """Stands in for a filesystem whose ``protocol`` is a list, as old s3fs reports."""

    protocol: ClassVar[list[str]] = ["s3", "s3a"]

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.reads: list[str] = []

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        pass

    def pipe_file(self, path: str, data: bytes) -> None:
        self.objects[path] = data

    def find(self, prefix: str) -> list[str]:
        return [p for p in self.objects if p.startswith(prefix + "/")]

    def glob(self, pattern: str) -> list[str]:
        return [p for p in self.objects if fnmatch.fnmatchcase(p, pattern)]

    def cat_ranges(self, paths: list[str], starts: list[Any], ends: list[Any], **_kw: Any) -> list[bytes]:
        self.reads.extend(paths)
        return [self.objects[p] for p in paths]


def test_completed_reads_each_record_once_with_a_list_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _ListProtocolFS()
    monkeypatch.setattr("fsspec.core.url_to_fs", lambda url, **_o: (fs, "bkt/root"))
    store = gcp.FsspecStore("s3://bkt/root")
    for i in range(3):
        store.record_done(f"t{i}", "p", f"{i:064d}")
        fs.objects[f"{store.objects}/{i:064d}"] = b""
    assert len(store.completed()) == 3
    assert len(store.completed()) == 3
    assert len(fs.reads) == 3
