"""m51 anchor G — the parquet-KV manifest key set, levels list, and the two round-trips (§6.4e/g).

The manifest travels in parquet key-value file metadata (greenfield — graphed writes none today). Its
top-level KEY SET is every label PLUS the reserved `"levels"` entry; `"levels"` is a LIST in §6.4e's
bound order `(depth, field_path or "")`, spelled literally (never a Python set — the levels entry
needs its own assertion because the round-trip supplies both levels itself and cannot discriminate a
reader that never consults it). The augmented file round-trips through `ak.from_parquet` (the arrow
write path must reproduce awkward's own KV entries), and an UNVARIED write keeps the `ak.to_parquet`
path, carries NO manifest, and is byte-identical to today — asserted SAME-PROCESS, never as a
committed `.parquet` fixture (§6.4g: a parquet footer embeds its writer version — R0.10a).
"""

from __future__ import annotations

import awkward as ak
import pytest
from m51_write_fixtures import (
    MANIFEST_KEY,
    events_context,
    multifield_skim_inputs,
    raw_manifest,
    raw_manifest_bytes,
    weight_skim_inputs,
)

import graphed.awkward as ga

pytest.importorskip("pyarrow")


def test_object_migration_manifest_key_set_and_levels_list(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask, jet_mask = multifield_skim_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "mf"), select={0: evt_mask, ("Jet", 1): jet_mask})  # type: ignore[call-arg]
    manifest = raw_manifest(paths[0])

    assert set(manifest) == {"nominal", "jes_up", "jes_down", "levels"}
    assert manifest["levels"] == [0, ["Jet", 1]]  # bare-depth-0 before field-scoped ("Jet", 1)


def test_weight_only_manifest_key_set_and_levels_list(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = weight_skim_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "w"), select={0: evt_mask})  # type: ignore[call-arg]
    manifest = raw_manifest(paths[0])

    assert set(manifest) == {"nominal", "murf_1", "murf_5em1", "murf_2", "levels"}
    assert manifest["levels"] == [0]


def test_augmented_file_roundtrips_through_ak_from_parquet(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask, jet_mask = multifield_skim_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "rt"), select={0: evt_mask, ("Jet", 1): jet_mask})  # type: ignore[call-arg]
    table = ak.from_parquet(paths[0])  # the arrow-write path must preserve awkward's own KV entries
    assert {"Jet", "MET"}.issubset(set(table.fields))


def test_unvaried_write_keeps_ak_to_parquet_bytes_and_writes_no_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session, events = events_context()
    jets = events.Jet
    paths = ga.to_parquet(jets, str(tmp_path / "plain"))  # unvaried: no select=, feature-present path
    reference = str(tmp_path / "ref.parquet")
    ak.to_parquet(session.materialize(jets), reference)

    with open(paths[0], "rb") as g, open(reference, "rb") as r:
        assert g.read() == r.read()  # SAME-PROCESS byte-golden (§6.4g), no committed blob
    assert raw_manifest_bytes(paths[0]) is None
    import pyarrow.parquet as pq  # noqa: PLC0415

    assert MANIFEST_KEY not in (pq.ParquetFile(paths[0]).metadata.metadata or {})
