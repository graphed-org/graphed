"""m51 anchor N — the augmented write is a SINGLE read pass (§5.2b witness for §7.1).

§7.1 forbids per-variation re-execution: one Session, one IR, one plan for nominal + all variations.
The augmented write stays one plan / one read pass — the §5.2b single-read witness applies to the
write run. A `PartitionedSource` that counts `read_partition` calls is the instrument: a varied write
over N labels and P partitions must read each partition ONCE (P reads), never once per label (P·N).
"""

from __future__ import annotations

import pytest
from m51_write_fixtures import counting_source_events

import graphed
import graphed.awkward as ga
from graphed.awkward import gak

pytest.importorskip("pyarrow")


def test_a_varied_write_reads_each_partition_exactly_once(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events, source = counting_source_events()
    jets = events.Jet
    up = gak.with_field(jets, jets.pt * 1.05, "pt")
    down = gak.with_field(jets, jets.pt * 0.95, "pt")
    ctx = graphed.vary(events, "jes", collections={"Jet": {"up": up, "down": down}})
    vjets = ctx.Jet
    evt_mask = gak.any(vjets.pt > 30.0, axis=1)
    jet_mask = vjets.pt > 25.0

    # the program under test really is varied (three universes), so a per-label read would be 3x
    assert len(graphed.labels(vjets)) == 3
    partitions = 3
    ga.to_parquet(vjets, str(tmp_path / "n"), select={0: evt_mask, 1: jet_mask}, steps_per_file=partitions)  # type: ignore[call-arg]

    assert len(source.reads) == partitions  # one read per partition — NOT partitions * labels
