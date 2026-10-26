"""A Session recording several datasets writes the one an array reads."""

from __future__ import annotations

import os

import awkward as ak
import pytest

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
