"""m51 anchor B — bit-exact per-universe round-trip through `read_varied` (§6.4a/c/e, §7.2).

Write an augmented skim, then reconstruct every universe's post-selection values and row set with
`graphed.awkward.read_varied` and compare BIT-FOR-BIT against an INDEPENDENT in-memory varied run
(plain awkward, from the same input events — §5.2a's self-derivation trap forbids taking the
reference from the graph under test). XOR deltas are exact by construction, so equality is exact.

Coverage items (the plan allows separate fixtures; each states the record shape it writes):
  * BARE-KEY single-collection Jet skim  `to_parquet(events.Jet, select={0: evt, 1: jet})` — the
    record IS the jagged Jet collection, so both levels take the RECORD'S OWN bare-depth keys; this
    is the mandatory bare-key coverage item (§6.4a's bare-`k` branch is new m51 source that would
    otherwise carry zero frozen-suite diff coverage);
  * FIELD-SCOPED multi-field record  `{Jet: var*{…}, MET: {…}}` written through
    `select={0: evt, ("Jet", 1): jet}` — object-level migration under a JES shift, the field-scoped
    per-level channel that a single row mask cannot express;
  * WEIGHT-ONLY μR/μF labels whose stored factor is `graphed.weight(c)` in the RECORD'S OWN row
    space (§6.4b) — carrying the e-canonical `murf_5em1` label verbatim AND the all-zero-delta
    REPLICATION discriminator (`murf_1` interns to nominal's node id, so `mark_output` de-dups and a
    POSITIONAL unpack in `_WritePart` would misassign every label after the collapse; §7.2 binds
    resolution BY NODE ID with replication).
"""

from __future__ import annotations

from typing import Any

import pytest
from m51_write_fixtures import (
    JES_LABELS,
    ROW_LABELS,
    as_list,
    eager_jet_universe,
    eager_multifield_universe,
    eager_row_universe,
    events_context,
    jet_skim_inputs,
    multifield_skim_inputs,
    weight_skim_inputs,
)

import graphed
import graphed.awkward as ga

pytest.importorskip("pyarrow")


def _one_part(paths: list[str]) -> str:
    assert len(paths) == 1, f"expected a single part with steps_per_file=1, got {paths}"
    return paths[0]


def test_bare_key_jet_skim_roundtrips_every_universe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events = events_context()
    record, evt_mask, jet_mask = jet_skim_inputs(events)
    dest = str(tmp_path / "jets")
    paths = ga.to_parquet(record, dest, select={0: evt_mask, 1: jet_mask})  # type: ignore[call-arg]
    got = ga.read_varied(_one_part(paths))  # type: ignore[attr-defined]

    assert tuple(got) == JES_LABELS  # every universe reconstructed, nominal first (§2.4 order)
    for label in JES_LABELS:
        assert as_list(got[label]) == as_list(eager_jet_universe(label)), label


def test_field_scoped_multifield_skim_roundtrips_every_universe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events = events_context()
    record, evt_mask, jet_mask = multifield_skim_inputs(events)
    dest = str(tmp_path / "mixed")
    paths = ga.to_parquet(record, dest, select={0: evt_mask, ("Jet", 1): jet_mask})  # type: ignore[call-arg]
    got = ga.read_varied(_one_part(paths))  # type: ignore[attr-defined]

    assert tuple(got) == JES_LABELS
    for label in JES_LABELS:
        assert as_list(got[label]) == as_list(eager_multifield_universe(label)), label


def _w_node_ids(record: Any) -> list[int]:
    return [graphed.universe(record, label).w.node_id for label in ROW_LABELS]


def test_weight_only_labels_replicate_the_collapsed_output_and_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events = events_context()
    record, evt_mask = weight_skim_inputs(events)

    # WITNESS the collapse the anchor turns on: `murf_1`'s stored factor interns to nominal's node id,
    # and it is NOT the last label — so `mark_output` de-dups it away and the value list is SHORTER
    # than the marked-output list, the exact condition under which a positional unpack diverges from
    # node-id resolution (§7.2). A suite run against a record where nothing collapsed would not test
    # replication at all, so this assertion keeps the fixture honest.
    node_ids = _w_node_ids(record)
    assert node_ids[ROW_LABELS.index("murf_1")] == node_ids[ROW_LABELS.index("nominal")]
    assert len(set(node_ids)) < len(node_ids)  # a real de-dup: fewer distinct outputs than labels

    dest = str(tmp_path / "weights")
    paths = ga.to_parquet(record, dest, select={0: evt_mask})  # type: ignore[call-arg]
    got = ga.read_varied(_one_part(paths))  # type: ignore[attr-defined]

    assert tuple(got) == ROW_LABELS  # e-canonical `murf_5em1` label survives verbatim
    # the collapsed label reconstructs to nominal (all-zero delta) AND the labels AFTER it reconstruct
    # to their OWN scaled values — a positional unpack gives `murf_5em1`/`murf_2` the wrong universe.
    for label in ROW_LABELS:
        assert as_list(got[label]) == as_list(eager_row_universe(label)), label
    assert as_list(got["murf_1"].w) == as_list(got["nominal"].w)  # the replication, explicitly
    assert as_list(got["murf_5em1"].w) != as_list(got["nominal"].w)  # not collapsed onto nominal
