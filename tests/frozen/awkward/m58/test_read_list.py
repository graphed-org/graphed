"""m58 integ-m58-H1/H2 — the source decides the read list every driver ships.

Three drivers compute a partition read list with the source DATA object in scope: `aggregate_plan`,
the plain `to_parquet` write and the `select=` varied write. Each asks a source that has a callable
`projected_columns` attribute, once per driver call, with the output `Array`s it was given, and
ships that answer verbatim; a source without the attribute keeps the driver's own computation and
stays a `PartitionedSource` (the Protocol gains no member).
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import pytest
from m58_declaration_fixtures import (
    AGGREGATE_COLUMNS,
    AGGREGATE_VALUE,
    DECLARED,
    PARQUET_COLUMNS,
    PARQUET_VALUE,
    VARIED_COLUMNS,
    VARIED_LABELS,
    VARIED_UNIVERSES,
    DeclaringSource,
    PlainSource,
    aggregate_over,
    partitioned,
    varied_record,
)

import graphed.awkward as ga
from graphed import Array
from graphed.core.execution import SequentialRunner
from graphed.write import PartitionedSource

pytest.importorskip("pyarrow")


def _read_back(paths: list[str]) -> ak.Array:
    return ak.concatenate([ak.from_parquet(path) for path in paths])


def test_the_aggregate_driver_ships_the_sources_declared_read_list() -> None:
    source = DeclaringSource()
    plan, outputs = aggregate_over(source)

    assert source.calls == 1  # driver-side, once per driver call
    assert len(source.outputs) == 2
    assert source.outputs[0] is outputs[0] and source.outputs[1] is outputs[1]  # identity AND order
    assert plan.process.columns == DECLARED  # the hook's list, as a tuple
    assert SequentialRunner().run(plan).value == AGGREGATE_VALUE
    assert source.seen == [DECLARED, DECLARED]  # every read_partition of that plan


def test_the_parquet_write_driver_ships_the_sources_declared_read_list(tmp_path: Path) -> None:
    source = DeclaringSource()
    _session, root = partitioned(source)
    output = root.x * 2.0

    paths = ga.to_parquet(output, str(tmp_path / "w"), steps_per_file=2)

    assert source.calls == 1
    assert len(source.outputs) == 1 and source.outputs[0] is output
    assert source.seen == [DECLARED, DECLARED]
    assert ak.to_list(_read_back(paths).data) == PARQUET_VALUE


def test_the_varied_write_driver_ships_the_sources_declared_read_list(tmp_path: Path) -> None:
    source = DeclaringSource()
    session, root = partitioned(source)
    record, mask = varied_record(root)

    paths = ga.to_parquet(record, str(tmp_path / "v"), select={0: mask}, steps_per_file=2)

    assert source.calls == 1
    # the varied driver's outputs are the marked record's own universe members, not a caller tuple
    assert source.outputs
    assert all(isinstance(out, Array) and out.session is session for out in source.outputs)
    assert source.seen == [DECLARED, DECLARED]
    universes = [ga.read_varied(path) for path in paths]
    assert tuple(sorted(universes[0])) == VARIED_LABELS
    for label, expected in VARIED_UNIVERSES.items():
        assert ak.to_list(ak.concatenate([part[label] for part in universes])) == expected


def test_a_source_without_the_hook_keeps_each_drivers_own_read_list(tmp_path: Path) -> None:
    aggregate = PlainSource()
    plan, _outputs = aggregate_over(aggregate)
    assert plan.process.columns == AGGREGATE_COLUMNS
    assert SequentialRunner().run(plan).value == AGGREGATE_VALUE
    assert aggregate.seen == [AGGREGATE_COLUMNS, AGGREGATE_COLUMNS]

    parquet = PlainSource()
    _session, root = partitioned(parquet)
    paths = ga.to_parquet(root.x * 2.0, str(tmp_path / "w"), steps_per_file=2)
    assert parquet.seen == [PARQUET_COLUMNS, PARQUET_COLUMNS]
    assert ak.to_list(_read_back(paths).data) == PARQUET_VALUE

    varied = PlainSource()
    _session, root = partitioned(varied)
    record, mask = varied_record(root)
    ga.to_parquet(record, str(tmp_path / "v"), select={0: mask}, steps_per_file=2)
    assert varied.seen == [VARIED_COLUMNS, VARIED_COLUMNS]


def test_the_partitioned_source_protocol_gains_no_member() -> None:
    class HookOnly:
        def projected_columns(self, outputs: object) -> tuple[str, ...]:
            return DECLARED

    # a source with NO hook must keep passing the isinstance check the drivers dispatch on
    assert isinstance(PlainSource(), PartitionedSource)
    assert isinstance(DeclaringSource(), PartitionedSource)
    assert not isinstance(HookOnly(), PartitionedSource)  # the hook never confers the protocol
