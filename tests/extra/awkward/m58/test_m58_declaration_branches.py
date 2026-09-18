"""m58 implementation-review closures: the branches the frozen suite states but cannot fail on.

Each leg kills a one-hunk mutant the whole frozen tree survives — the three drivers' short-circuit
(the own column computation hoisted back out of its `is None` guard), an empty declaration read as
"no answer", `callable()` dropped from the hook lookup, and the External stand-in's `AwkwardForm`
guard dropped.
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
