"""The numpy half of the two-source refusal and of ``graphed.join``'s two routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.numpy as gn
from graphed.core import Partition
from graphed.errors import SourceRowsError, StageError
from graphed.numpy import NumpyBackend
from graphed.numpy.forms import NumpyForm
from graphed.numpy.io import from_parquet


@pytest.fixture
def reads(tmp_path: Path) -> tuple[Any, Any]:
    pytest.importorskip("pyarrow")
    session = graphed.Session(NumpyBackend())
    tables = {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}
    for name, xs in tables.items():
        ak.to_parquet(ak.Array({"x": xs}), str(tmp_path / f"{name}.parquet"))
    a = from_parquet("a", str(tmp_path / "a.parquet"), session=session)
    b = from_parquet("b", str(tmp_path / "b.parquet"), session=session)
    return a["x"], b["x"]


def test_two_parquet_reads_are_two_row_sets(reads: tuple[Any, Any]) -> None:
    a, b = reads
    with pytest.raises(SourceRowsError, match="'a' and 'b'"):
        a + b


def test_what_consumed_the_row_axis_crosses_sources(reads: tuple[Any, Any]) -> None:
    a, b = reads
    assert graphed.materialize(a.sum() + b.sum()) == 15.0
    counts = [np.histogram(x, bins=2, range=(0.0, 4.0))[0] for x in reads]
    assert np.array_equal(graphed.materialize(counts[0] + counts[1]), [1, 5])


def test_a_set_test_reads_the_other_source_as_a_set(reads: tuple[Any, Any]) -> None:
    a, b = reads
    assert np.array_equal(graphed.materialize(a[np.isin(a, b)]), [2.0, 3.0])
    assert np.array_equal(graphed.materialize(b + np.searchsorted(a, b)), [3.0, 5.0, 7.0])
    with pytest.raises(SourceRowsError, match="'a' and 'b'"):
        b[np.isin(a, b)]
    with pytest.raises(SourceRowsError, match="'a' and 'b'"):
        a + np.searchsorted(a, b)


class _Reader:
    """A partitioned source that is no loader, as a ROOT reader is not."""

    def partitions(self, steps_per_file: int) -> tuple[Partition, ...]:
        return (Partition("reader", "", 0, 1),)

    def read_partition(self, partition: Partition, columns: object, resources: object) -> object:
        return np.arange(1.0)


def test_a_partitioned_reader_is_a_row_set() -> None:
    session = graphed.Session(NumpyBackend())
    form = NumpyForm(np.dtype(float))
    a = session.source("a", form=form, data=_Reader())
    b = session.source("b", form=form, data=_Reader())
    with pytest.raises(SourceRowsError, match="'a' and 'b'"):
        a + b


def test_concatenating_an_inner_axis_pairs_rows() -> None:
    session = graphed.Session(NumpyBackend())
    form = NumpyForm(np.dtype(float), shape=(None, 2))
    a = session.source("a", form=form, data=_Reader())
    b = session.source("b", form=form, data=_Reader())
    assert session.reachable_sources(np.concatenate([a, b]).node_id) == {a.node_id, b.node_id}
    with pytest.raises(SourceRowsError, match="'a' and 'b'"):
        np.concatenate([a, b], axis=1)


def test_in_memory_arrays_stay_pairable() -> None:
    session = graphed.Session(NumpyBackend())
    a = gn.from_array("a", np.arange(3.0), session=session)
    b = gn.from_array("b", np.arange(3.0) * 10.0, session=session)
    assert np.array_equal(graphed.materialize(a + b), [0.0, 11.0, 22.0])
    assert graphed.materialize(a.sum() + b.sum()) == 33.0


def test_positional_join_of_record_sources() -> None:
    session = graphed.Session(NumpyBackend())
    a = gn.from_record("a", x=np.arange(3.0), session=session)
    b = gn.from_record("b", y=np.arange(3.0) * 2.0, session=session)
    joined = graphed.join(a, b, how="positional")
    assert np.array_equal(graphed.materialize(joined["x"] + joined["y"]), [0.0, 3.0, 6.0])


def test_positional_join_refuses_a_one_row_side() -> None:
    session = graphed.Session(NumpyBackend())
    a = gn.from_record("a", x=np.arange(3.0), session=session)
    one = gn.from_record("one", y=np.arange(1.0), session=session)
    with pytest.raises(StageError, match="3 rows against 1"):
        graphed.materialize(graphed.join(a, one, how="positional")["y"])


def test_a_keyed_join_matches_event_numbers_too_wide_to_pack() -> None:
    session = graphed.Session(NumpyBackend())
    wide = np.array([2**40, 2**40 + 2**20, 5], dtype=np.uint64)
    a = gn.from_record(
        "a", run=np.ones(3, np.int64), event=wide, x=np.arange(3.0), session=session
    )
    b = gn.from_record(
        "b",
        run=np.ones(3, np.int64),
        event=wide[::-1],
        y=np.array([2.0, 1.0, 0.0]),
        session=session,
    )
    joined = graphed.join(a, b, on=["run", "event"])
    got = graphed.materialize(joined["x"] - joined["y"])
    assert np.array_equal(got, [0.0, 0.0, 0.0])


def test_public_pack_key_still_refuses_a_wide_value() -> None:
    session = graphed.Session(NumpyBackend())
    a = gn.from_record("a", event=np.array([2**20], dtype=np.uint64), session=session)
    with pytest.raises(StageError, match=r"2\*\*20"):
        graphed.materialize(graphed.pack_key(a, on=["event"]))
