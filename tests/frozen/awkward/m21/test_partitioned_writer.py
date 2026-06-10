"""M21: the generic writer dispatches on the PartitionedSource protocol (P3.6 revision).

`graphed_awkward.io.to_parquet` must write ANY deferred array partition-wise when its source data
implements `graphed.write.PartitionedSource` — with the source's whole-dataset loader NEVER
invoked (the efficiency witness: counters), and a read list that merges the graph's SYNTACTIC
source-field accesses (evaluation replays every node — a zip's untouched legs included) with the
buffer projection's leaf refinement (DATA leaves; a carrier leaf for offsets-only needs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import awkward as ak
import pytest
from graphed import Session

pytest.importorskip("pyarrow")

from graphed.write import PartitionedSource, SequentialRunner
from graphed_core import Partition
from graphed_core.execution import Plan

import graphed_awkward.io as gio
from graphed_awkward import AwkwardBackend, AwkwardForm, from_parquet, gak

EVENTS = ak.Array(
    {
        "Jet": [[{"pt": 50.0, "eta": 0.5}], [], [{"pt": 70.0, "eta": 2.1}, {"pt": 20.0, "eta": 0.0}]],
        "x": [1.0, 2.0, 3.0],
    }
)


@dataclass
class ChunkedToySource:
    """A protocol source over an in-memory array, with counters as the efficiency witnesses."""

    data: ak.Array
    whole_calls: list = field(default_factory=list)
    part_reads: list = field(default_factory=list)

    def __call__(self) -> ak.Array:  # the whole-dataset loader (must NEVER run during writes)
        self.whole_calls.append(1)
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://chunks", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources) -> ak.Array:  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        chunk = self.data[part.entry_start : part.entry_stop]
        return chunk[list(columns)] if columns else chunk


def _toy_session() -> tuple[Session, object, ChunkedToySource]:
    s = Session(AwkwardBackend())
    toy = ChunkedToySource(EVENTS)
    tt = ak.Array(EVENTS.layout.to_typetracer(forget_length=True))
    return s, s.source("events", form=AwkwardForm(tt), data=toy), toy


def test_protocol_sources_write_partitionwise_without_materializing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, g, toy = _toy_session()
    expr = g.x * 2.0
    paths = gio.to_parquet(expr, os.path.join(tmp_path, "out"), steps_per_file=3)
    assert isinstance(paths, list) and len(paths) == 3
    assert toy.whole_calls == []  # the whole-dataset loader NEVER ran — no big materialize
    assert len(toy.part_reads) == 3  # one read per partition, tiling the dataset
    assert sorted(toy.part_reads) == [(0, 1), (1, 2), (2, 3)]
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert ak.array_equal(back["data"], EVENTS.x * 2.0)


def test_protocol_partitions_are_blind_in_the_disabled_plan(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, g, toy = _toy_session()
    plan = gio.to_parquet(g.x + 1.0, os.path.join(tmp_path, "o"), steps_per_file=2, compute=False)
    assert isinstance(plan, Plan)
    assert all(t.partition.is_blind for t in plan.tasks)  # planning opened nothing (R7.9)
    assert toy.whole_calls == [] and toy.part_reads == []  # and read nothing
    later = SequentialRunner().run(plan).value
    back = ak.concatenate([ak.from_parquet(p) for p in later])
    assert ak.array_equal(back["data"], EVENTS.x + 1.0)


def test_parquet_dataset_loader_implements_the_protocol(tmp_path) -> None:  # type: ignore[no-untyped-def]
    where = os.path.join(tmp_path, "ds")
    os.makedirs(where)
    ak.to_parquet(EVENTS, os.path.join(where, "e-0.parquet"))
    s = Session(AwkwardBackend())
    g = from_parquet(s, "events", where)
    ((_nid, data),) = s.sources().items()
    assert isinstance(data, PartitionedSource)  # one code path: parquet rides the same protocol
    plan = gio.to_parquet(g.x * 3.0, os.path.join(tmp_path, "o"), compute=False)
    assert all(t.partition.is_blind for t in plan.tasks)  # type: ignore[union-attr]


def test_read_list_merges_syntactic_access_with_buffer_refinement(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # NOTE: one session per compiled expression — graph outputs ACCUMULATE per session, so
    # compiling two different expressions from one session yields a multi-output IR (a recorded
    # compile_ir footgun, out of this milestone's scope)
    where = os.path.join(tmp_path, "ds")
    os.makedirs(where)
    ak.to_parquet(EVENTS, os.path.join(where, "e-0.parquet"))

    # offsets-only need: the syntactic access is Jet; the buffer view refines to ONE carrier leaf
    s1 = Session(AwkwardBackend())
    g1 = from_parquet(s1, "events", where)
    expr = gak.num(g1.Jet, axis=1) + g1.x
    plan = gio.to_parquet(expr, os.path.join(tmp_path, "o1"), compute=False)
    assert set(plan.process.columns) == {"Jet.eta", "x"}  # type: ignore[union-attr, attr-defined]

    # the zip finding: evaluation REPLAYS the untouched leg, so its field must still be read
    s2 = Session(AwkwardBackend())
    g2 = from_parquet(s2, "events", where)
    rec = gak.zip({"a": g2.x, "b": gak.num(g2.Jet, axis=1)})
    out = rec.a  # only a's DATA is touched, but the b leg (Jet's offsets) is replayed
    plan2 = gio.to_parquet(out, os.path.join(tmp_path, "o2"), compute=False)
    assert set(plan2.process.columns) == {"Jet.eta", "x"}  # type: ignore[union-attr, attr-defined]
    paths = gio.to_parquet(out, os.path.join(tmp_path, "o3"))
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert ak.array_equal(back["data"], EVENTS.x)
