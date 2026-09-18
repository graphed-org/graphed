"""m58 implementation-review closures: the branches the frozen suite states but cannot fail on.

Each leg kills a one-hunk mutant the whole frozen tree survives — the three drivers' short-circuit
(the own column computation hoisted back out of its `is None` guard), an empty declaration read as
"no answer" at each write driver (`self.columns` re-spelled `self.columns or None`, or the driver's
guard re-spelled `if not columns:`), the hook's argument left un-tupled, `callable()` dropped from
the hook lookup, and the External stand-in's `AwkwardForm` guard dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from m58_declaration_fixtures import (
    AGGREGATE_COLUMNS,
    AGGREGATE_VALUE,
    DECLARED,
    DeclaringSource,
    PlainSource,
    aggregate_over,
    declared_external,
    external_session,
    partitioned,
    varied_record,
)

import graphed
import graphed.awkward as ga
from graphed.awkward import AwkwardForm
from graphed.awkward import io as awkward_io
from graphed.core.execution import SequentialRunner

pytest.importorskip("pyarrow")


def _explode(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a declared read list must short-circuit the driver's own computation")


def test_a_declaration_short_circuits_the_aggregate_drivers_own_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graphed.aggregate, "read_columns", _explode)

    with pytest.raises(AssertionError):  # the patched site is the one this driver reaches
        aggregate_over(PlainSource())

    plan, _outputs = aggregate_over(DeclaringSource())
    assert plan.process.columns == DECLARED


def test_a_declaration_short_circuits_the_parquet_write_drivers_own_computation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(awkward_io, "_evaluation_columns", _explode)
    _plain_session, plain_root = partitioned(PlainSource())

    with pytest.raises(AssertionError):
        ga.to_parquet(plain_root.x * 2.0, str(tmp_path / "control"), steps_per_file=2)

    source = DeclaringSource()
    _session, root = partitioned(source)
    ga.to_parquet(root.x * 2.0, str(tmp_path / "declared"), steps_per_file=2)
    assert source.seen == [DECLARED, DECLARED]


def test_a_declaration_short_circuits_the_varied_write_drivers_own_computation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(awkward_io, "_evaluation_columns_union", _explode)
    _plain_session, plain_root = partitioned(PlainSource())
    plain_record, plain_mask = varied_record(plain_root)

    with pytest.raises(AssertionError):
        ga.to_parquet(plain_record, str(tmp_path / "control"), select={0: plain_mask}, steps_per_file=2)

    source = DeclaringSource()
    _session, root = partitioned(source)
    record, mask = varied_record(root)
    ga.to_parquet(record, str(tmp_path / "declared"), select={0: mask}, steps_per_file=2)
    assert source.seen == [DECLARED, DECLARED]


def test_an_empty_declaration_ships_an_empty_read_list() -> None:
    source = DeclaringSource(answer=())

    plan, _outputs = aggregate_over(source)

    assert source.calls == 1
    assert plan.process.columns == ()  # an answer of "nothing", not the absence of an answer


def _reads_everything(source: Any) -> Any:
    """Keeps recording the read list it is handed, then reads the whole chunk anyway, so a
    declaration of "nothing" still lets the write finish. An instance attribute rather than a
    subclass: the frozen fixture module is `Any` to mypy, which cannot be subclassed."""

    def read_partition(partition: Any, columns: Any, _resources: Any) -> Any:
        source.seen.append(columns)
        part = partition.resolve(len(source.data))
        return source.data[part.entry_start : part.entry_stop]

    source.read_partition = read_partition
    return source


def test_the_parquet_write_driver_ships_an_empty_declaration_to_every_read(tmp_path: Path) -> None:
    source = _reads_everything(DeclaringSource(answer=()))
    _session, root = partitioned(source)

    ga.to_parquet(root.x * 2.0, str(tmp_path / "empty"), steps_per_file=2)

    assert source.seen == [(), ()]  # "read nothing", not the driver's "everything" sentinel


def test_the_varied_write_driver_ships_an_empty_declaration_to_every_read(tmp_path: Path) -> None:
    source = _reads_everything(DeclaringSource(answer=()))
    _session, root = partitioned(source)
    record, mask = varied_record(root)

    ga.to_parquet(record, str(tmp_path / "empty"), select={0: mask}, steps_per_file=2)

    assert source.seen == [(), ()]


def test_a_hook_less_whole_record_write_still_reads_with_the_sources_own_selection(
    tmp_path: Path,
) -> None:
    """The control for the sentinel translation: with no declaration, a write whose own column
    computation answers "everything" must still reach `read_partition` as `None`."""
    source = PlainSource()
    _session, root = partitioned(source)

    ga.to_parquet(root, str(tmp_path / "whole"), steps_per_file=2)

    assert source.seen == [None, None]


def test_a_hook_less_varied_whole_record_write_also_reads_with_the_sources_own_selection(
    tmp_path: Path,
) -> None:
    """The varied end of that control: a getitem on the source is a non-field op, so the union
    absorbs to its own "everything" — which must also reach `read_partition` as `None`."""
    source = PlainSource()
    _session, root = partitioned(source)
    record, mask = varied_record(root[root.z > 8.0])

    ga.to_parquet(record, str(tmp_path / "whole"), select={0: mask}, steps_per_file=2)

    assert source.seen == [None, None]


def _asserts_a_tuple(source: Any) -> Any:
    """Refuses anything but a tuple of outputs; only the varied driver holds its own outputs in a
    list, so that is where an un-tupled hand-off shows."""
    hook = source.projected_columns

    def projected_columns(outputs: Any) -> Any:
        assert type(outputs) is tuple, f"the hook takes a tuple, got {type(outputs).__name__}"
        return hook(outputs)

    source.projected_columns = projected_columns
    return source


def test_the_varied_write_driver_hands_the_hook_a_tuple(tmp_path: Path) -> None:
    source = _asserts_a_tuple(DeclaringSource())
    _session, root = partitioned(source)
    record, mask = varied_record(root)

    ga.to_parquet(record, str(tmp_path / "tupled"), select={0: mask}, steps_per_file=2)

    assert source.calls == 1
    assert source.seen == [DECLARED, DECLARED]


def test_a_non_callable_projected_columns_attribute_is_not_a_declaration() -> None:
    source = PlainSource()
    source.projected_columns = DECLARED  # data under that name, not a hook

    plan, _outputs = aggregate_over(source)

    assert plan.process.columns == AGGREGATE_COLUMNS  # the driver's own list
    assert SequentialRunner().run(plan).value == AGGREGATE_VALUE


@dataclass(frozen=True)
class _ForeignForm:
    """A form recorded by another package (graphed-histogram's `HistogramForm` is one): it
    describes a value that is not an array at all, so it has no typetracer to stand in with."""

    def describe(self) -> str:
        return "m58-foreign"


def test_an_external_with_a_non_awkward_recorded_form_stands_in_with_its_first_input() -> None:
    session, root = external_session()
    node = declared_external(session, [root.z], _ForeignForm(), "foreign")

    recorded = session.form(node)
    assert not isinstance(recorded, AwkwardForm) and not hasattr(recorded, "tt")

    assert ga.project(node, on_fail="pass").read_columns == {"events": frozenset({"z"})}
