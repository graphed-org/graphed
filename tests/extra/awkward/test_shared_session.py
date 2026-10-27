"""A Session recording several datasets writes the one an array reads."""

from __future__ import annotations

import os
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import GraphedError, Session

pytest.importorskip("pyarrow")
import graphed.awkward.io as gio
from graphed.awkward import AwkwardBackend, from_awkward, from_parquet, gak


def test_to_parquet_writes_the_source_its_array_reads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    where = os.path.join(tmp_path, "events.parquet")
    ak.to_parquet(ak.Array({"x": [1.0, 2.0, 3.0]}), where)
    s = Session(AwkwardBackend())
    from_awkward(s, "other", ak.Array({"x": [9.0]}))
    g = from_parquet(s, "events", where)
    paths = gio.to_parquet(g.x * 2.0, os.path.join(tmp_path, "out"))
    assert isinstance(paths, list)
    assert ak.concatenate([ak.from_parquet(p) for p in paths])["data"].to_list() == [2.0, 4.0, 6.0]


def test_from_awkward_of_one_array_is_one_slot_and_of_another_is_refused() -> None:
    s = Session(AwkwardBackend())
    events = ak.Array({"x": [1.0, 2.0]})
    assert from_awkward(s, "events", events).node_id == from_awkward(s, "events", events).node_id
    with pytest.raises(ValueError, match="already bound to other data"):
        from_awkward(s, "events", ak.Array({"x": [3.0, 4.0]}))


def test_a_grouped_join_across_sessions_points_at_one_session() -> None:
    left = from_awkward(Session(AwkwardBackend()), "left", ak.Array({"k": [1]}))
    right = from_awkward(Session(AwkwardBackend()), "right", ak.Array({"k": [1]}))
    with pytest.raises(GraphedError, match="same Session; record both into one"):
        gak.join(left, right, on=["k"], grouped=True)


def test_materialize_reads_only_the_columns_its_arrays_use(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    paths = []
    for i in range(2):
        paths.append(os.path.join(tmp_path, f"events{i}.parquet"))
        ak.to_parquet(
            ak.Array({"pt": [[1.0 + i, 2.0], [3.0]], "eta": [[0.1], []], "phi": [[0.0], []]}),
            paths[-1],
        )
    s = Session(AwkwardBackend())
    events = from_parquet(s, "events", paths)
    (loader,) = s.sources().values()
    kind: Any = type(loader)
    reads: list[object] = []
    read_partition, load_all = kind.read_partition, kind.__call__

    def logged_read(self: Any, partition: Any, columns: Any, resources: Any) -> Any:
        reads.append(columns)
        return read_partition(self, partition, columns, resources)

    def logged_load(self: Any) -> Any:
        reads.append("all")
        return load_all(self)

    monkeypatch.setattr(kind, "read_partition", logged_read)
    monkeypatch.setattr(kind, "__call__", logged_load)
    other = from_awkward(Session(AwkwardBackend()), "other", ak.Array({"x": [5.0]}))

    got = graphed.materialize(
        {"sum": gak.sum(events.pt), "n": gak.num(events.eta, axis=1), "x": other.x}
    )
    assert (got["sum"], got["n"].to_list(), got["x"].to_list()) == (13.0, [1, 0, 1, 0], [5.0])
    assert reads == [("eta", "pt"), ("eta", "pt")]

    reads.clear()
    assert graphed.materialize(gak.num(events, axis=0)) == 4  # the whole record: every column
    assert reads == ["all"]
