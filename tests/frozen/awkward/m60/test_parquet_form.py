"""m60 integ-m60-P — a parquet source keeps awkward's record parameters.

`ak.from_arrow_schema` of a file's arrow schema drops everything awkward stored beside the arrow
types — record names and node parameters — so a deferred read of a file awkward WROTE does not
type like an eager read of it, and a behavior keyed on a record name never resolves. The form the
source records is what awkward's own reader gives a zero-row file of that schema; a file written
without awkward's metadata is unchanged, and no event data is read to find out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq
from m60_parquet_fixtures import (
    BEHAVIOR,
    FOREIGN_FORM,
    RECORD_NAME,
    TOTALS,
    awkward_dataset,
    eager,
    pyarrow_dataset,
    read_guard,
    tracer,
)

import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend


def _recorded_form(session: Session, array: Any) -> Any:
    return session.form(array).tt.layout.form


def test_the_recorded_form_equals_what_eager_awkward_gives_the_same_file(tmp_path: Path) -> None:
    (path,) = awkward_dataset(tmp_path)
    session = Session(AwkwardBackend())

    events = ga.from_parquet(session, "ev", path)

    assert _recorded_form(session, events) == eager(path).layout.form


def test_a_behavior_keyed_on_the_record_name_resolves_on_the_deferred_array(tmp_path: Path) -> None:
    (path,) = awkward_dataset(tmp_path)
    session = Session(AwkwardBackend(behavior=BEHAVIOR))

    events = ga.from_parquet(session, "ev", path)
    total = events.p.total

    assert session.form(total).describe() == str(ak.Array(tracer(eager(path), BEHAVIOR).p.total).type)
    assert ak.to_list(session.materialize(total)) == TOTALS
    assert ak.to_list(ak.Array(eager(path).layout, behavior=BEHAVIOR).p.total) == TOTALS


def test_columns_still_selects_and_the_selection_keeps_the_record_name(tmp_path: Path) -> None:
    (path,) = awkward_dataset(tmp_path)
    session = Session(AwkwardBackend())

    events = ga.from_parquet(session, "ev", path, columns=["p"])

    assert session.form(events).tt.fields == ["p"]
    assert RECORD_NAME in session.form(events).describe()


def test_a_file_without_awkwards_metadata_gets_the_form_it_gets_today(tmp_path: Path) -> None:
    path = pyarrow_dataset(tmp_path)
    session = Session(AwkwardBackend())

    events = ga.from_parquet(session, "ev", path)

    assert session.form(events).describe() == FOREIGN_FORM


def test_recording_a_source_reads_no_event_data_and_opens_only_the_first_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = awkward_dataset(tmp_path, parts=2)
    opened = read_guard(monkeypatch, paths)
    session = Session(AwkwardBackend())

    events = ga.from_parquet(session, "ev", list(paths), open_files=False)

    assert session.form(events).tt.fields == ["p", "o", "r"]
    assert [path for path in opened if path in paths] == [paths[0]]
    # the guard is a LIVE instrument: reading the same file's data now trips it
    with pytest.raises(AssertionError, match="event data read at record time"):
        ak.from_parquet(paths[0])


def test_a_zero_row_file_of_the_datasets_schema_stays_readable_under_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route the contract names must survive its own witness: the ban is path-scoped, so the
    dataset's schema is readable and a zero-row file written elsewhere reads back normally."""
    (path,) = awkward_dataset(tmp_path)
    read_guard(monkeypatch, [path])

    scratch = str(tmp_path / "scratch" / "empty.parquet")
    (tmp_path / "scratch").mkdir()
    pq.write_table(pq.ParquetFile(path).schema_arrow.empty_table(), scratch)

    assert ak.from_parquet(scratch).layout.form.fields == ["p", "o", "r"]
    with pytest.raises(AssertionError, match="event data read at record time"):
        pq.ParquetFile(path).read()
