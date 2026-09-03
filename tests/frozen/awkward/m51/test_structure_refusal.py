"""m51 anchor H — the §6.4d structure refusal (negative), with the object-migration positive control.

Deltas require same-shaped buffers. §2.1's construction check is a TYPE check (`var * float64`
matches whatever the per-event counts), so a variation that legitimately CHANGES multiplicity
(shift-dependent cleaning, overlap removal, a matched collection) passes §2.1 and then has no
representable XOR delta. Binding refusal: a stored varied field whose per-label OFFSETS differ from
nominal's is refused with an error naming the LABEL and the FIELD — offsets are data, so it surfaces
per partition at execution time (trap ledger §7). The supported v1 model is same-multiplicity
variation plus per-label validity masks, which STILL writes and round-trips — the positive control
that keeps object-level migration (a JES shift moving jets across a per-jet cut) a first-class case.
"""

from __future__ import annotations

from typing import Any

import pytest
from m51_write_fixtures import (
    JES_LABELS,
    as_list,
    eager_jet_universe,
    events_context,
    jet_skim_inputs,
)

import graphed
import graphed.awkward as ga
from graphed.awkward import gak
from graphed.errors import GraphedError

pytest.importorskip("pyarrow")


def _cleaning_record(events: Any) -> tuple[Any, Any]:
    """A `{Jet, MET}` record whose Jet field drops a DIFFERENT number of jets per universe (a
    shift-dependent cleaning): `clean_up`'s offsets differ from nominal's, `MET` is unvaried."""
    jets = events.Jet
    up = jets[jets.pt > 10.0]
    down = jets[jets.pt > 5.0]
    ctx = graphed.vary(events, "clean", collections={"Jet": {"up": up, "down": down}})
    record = gak.zip({"Jet": ctx.Jet, "MET": events.MET}, depth_limit=1)
    return record, gak.num(ctx.Jet) >= 0  # a trivially-true flat level-0 mask


def test_multiplicity_changing_field_is_refused_naming_the_label_and_field(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = _cleaning_record(events)
    with pytest.raises(GraphedError) as excinfo:
        ga.to_parquet(record, str(tmp_path / "clean"), select={0: evt_mask})  # type: ignore[call-arg]
    message = str(excinfo.value)
    assert "clean_up" in message  # the offending label (clean_down keeps nominal's offsets)
    assert "Jet" in message  # ...and the field whose per-label offsets diverged


def test_same_multiplicity_object_migration_still_writes_and_roundtrips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The positive control: a JES shift keeps every universe's jet OFFSETS equal to nominal's and
    expresses the migration as per-label inner masks — so it writes and round-trips (§6.4d)."""
    _s, events = events_context()
    record, evt_mask, jet_mask = jet_skim_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "ok"), select={0: evt_mask, 1: jet_mask})  # type: ignore[call-arg]
    got = ga.read_varied(paths[0])  # type: ignore[attr-defined]
    assert tuple(got) == JES_LABELS
    for label in JES_LABELS:
        assert as_list(got[label]) == as_list(eager_jet_universe(label)), label
