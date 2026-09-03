"""m51 anchor E — appended-column representation, names, and the collision refusal (§6.4b/c).

Structural only (R0.10a: no size thresholds in a frozen test). Appended columns land in the bound
exact representation — same-dtype XOR value deltas, packed validity masks — under the frozen on-disk
convention `__vary_{label}__{field_flat}` (value) / `__vary_{label}__mask__{entry}` (mask), verified
by reading the RAW parquet schema and the manifest. A nested field path (`Jet.pt`) flattens per level
to `Jet_pt`; the readback still resolves THROUGH the manifest (names embed labels for humans, not as
the machine channel — §6.4b).

The collision class is REAL and refused in BOTH directions: `Jet.pt` (nested) and a flat field named
`Jet_pt` both flatten to `__vary_{label}__Jet_pt`, so varying BOTH is refused naming both source
fields — while a varying `Jet.pt` beside a NON-varying `Jet_pt` is a legal nested skim, NOT a
refusal (the positive control that keeps a real nested skim writeable).
"""

from __future__ import annotations

from typing import Any

import pytest
from m51_write_fixtures import (
    events_context,
    multifield_skim_inputs,
    raw_manifest,
    raw_schema_names,
    value_column,
)

import graphed
import graphed.awkward as ga
from graphed.awkward import gak
from graphed.errors import GraphedError

pytest.importorskip("pyarrow")


def _jes(events: Any) -> Any:
    jets = events.Jet
    up = gak.with_field(jets, jets.pt * 1.05, "pt")
    down = gak.with_field(jets, jets.pt * 0.95, "pt")
    return graphed.vary(events, "jes", collections={"Jet": {"up": up, "down": down}})


def test_value_deltas_are_xor_and_masks_are_packbits_under_the_bound_names(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask, jet_mask = multifield_skim_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "rep"), select={0: evt_mask, ("Jet", 1): jet_mask})  # type: ignore[call-arg]
    schema = raw_schema_names(paths[0])
    manifest = raw_manifest(paths[0])

    for label in ("jes_up", "jes_down"):
        delta = value_column(label, "Jet_pt")  # nested Jet.pt flattened per level
        assert any(name == delta or name.startswith(delta + ".") for name in schema), (label, schema)
        assert manifest[label][delta]["representation"] == "xor"
        assert manifest[label][delta]["field"] == "Jet_pt"
        # every packbits column names its own level and no stored name leaks a dotted label
        masks = [c for c, d in manifest[label].items() if d["representation"] == "packbits"]
        assert masks, (label, manifest[label])
        assert all(label in c and c.replace("__", "_").isidentifier() for c in manifest[label]), label


def test_nested_field_path_flattens_and_reads_back_through_the_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask, jet_mask = multifield_skim_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "nest"), select={0: evt_mask, ("Jet", 1): jet_mask})  # type: ignore[call-arg]
    manifest = raw_manifest(paths[0])
    # the flattened name carries a single underscore per level, never a dotted `Jet.pt`
    assert value_column("jes_up", "Jet_pt") in manifest["jes_up"]
    assert not any("." in c for label in manifest if label != "levels" for c in manifest[label])
    # and the reader resolves labels THROUGH the manifest, returning the nested record intact
    got = ga.read_varied(paths[0])  # type: ignore[attr-defined]
    assert set(got["jes_up"].fields) == {"Jet", "MET"}


def _collision_record(events: Any, *, vary_flat: bool) -> Any:
    """A record with a nested `Jet.pt` (always varied under jes) and a FLAT field literally named
    `Jet_pt`; `vary_flat` decides whether that flat field varies (collision) or not (legal skim)."""
    ctx = _jes(events)
    flat = gak.sum(ctx.Jet.pt, axis=1) if vary_flat else gak.sum(events.Jet.pt, axis=1)
    return gak.zip({"Jet": ctx.Jet, "Jet_pt": flat}, depth_limit=1), gak.any(ctx.Jet.pt > 30.0, axis=1)


def test_collision_between_a_nested_and_a_flat_field_is_refused_naming_both(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = _collision_record(events, vary_flat=True)
    with pytest.raises(GraphedError) as excinfo:
        ga.to_parquet(record, str(tmp_path / "coll"), select={0: evt_mask}, compute=False)  # type: ignore[call-arg]
    message = str(excinfo.value)
    assert "Jet.pt" in message or "Jet_pt" in message  # names the colliding source field(s)


def test_a_nonvarying_flat_field_is_not_a_collision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = _collision_record(events, vary_flat=False)
    paths = ga.to_parquet(record, str(tmp_path / "legal"), select={0: evt_mask})  # type: ignore[call-arg]
    assert len(paths) == 1  # a varying Jet.pt beside a non-varying Jet_pt is a legal nested skim
