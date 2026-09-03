"""m51 extra witnesses (impl-review MUT-M5e MED + malformed-key LOW): two ``select=`` guardrails the
frozen suite happens not to drive. Self-contained (the frozen ``m51_write_fixtures`` live in a sibling
tree that is not on this process's path), built from ``graphed_corpus`` + the public graphed API.

  * MED — the level-≥1 STRUCTURAL mask-offsets refusal (``_VariedWritePart.__call__``): a field-scoped
    ``("Jet", 1)`` mask with the RIGHT depth but per-row offsets that do NOT match the Jet field (a
    Muon-shaped mask) clears the record-time depth check (2c) and lineage (2a), then must raise at
    EXECUTION time naming the offsets. Without the guard a wrong-shaped mask is stored against the
    field → silently wrong per-universe objects. Positive control: the matched per-jet mask writes.
  * LOW — the malformed ``select=`` key guard (``_normalize_select``): a key that is neither a bare
    depth ``int`` nor a ``(field_name, depth>=1)`` tuple is refused at the call.

Extra (non-frozen): both guards are load-bearing but the frozen anchors leave the firing direction
unwitnessed (see the m51 impl-review kill-matrix, MUT-M5e).
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest
from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.core.execution import Plan
from graphed.errors import GraphedError

pytest.importorskip("pyarrow")


def _write(record: Any, dest: str, select: Any, *, compute: bool = True) -> Any:
    return ga.to_parquet(record, dest, select=select, compute=compute)  # type: ignore[call-arg]


def _multifield() -> tuple[Any, Any, Any, Any]:
    """A ``{Jet, MET}`` record over a jes context, plus the level-0 event mask, the CORRECT per-jet
    (level-1) mask, and a WRONG per-muon (level-1) mask read through the SAME context (so it clears
    lineage). Muon's per-event counts differ from Jet's, so its offsets diverge from the Jet field."""
    session = Session(AwkwardBackend())
    root = from_awkward(session, "events", ak.Array(make_events(n_events=60, seed=51)))
    events = ga.gnano.events(root)
    jets = events.Jet
    ctx = graphed.vary(
        events,
        "jes",
        collections={
            "Jet": {
                "up": gak.with_field(jets, jets.pt * 1.05, "pt"),
                "down": gak.with_field(jets, jets.pt * 0.95, "pt"),
            }
        },
    )
    record = gak.zip({"Jet": ctx.Jet, "MET": events.MET}, depth_limit=1)
    evt_mask = gak.any(ctx.Jet.pt > 30.0, axis=1)
    jet_mask = ctx.Jet.pt > 25.0  # matched Jet offsets (positive control)
    muon_mask = ctx.Muon.pt > -1.0  # trivially-true per-muon mask: ndim-2, Muon offsets != Jet
    return record, evt_mask, jet_mask, muon_mask


def test_level1_mask_with_wrong_offsets_is_refused_at_execution_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A ``("Jet", 1)`` mask carrying MUON offsets is ndim-2, so it clears the record-time depth check
    and lineage — a ``Plan`` is returned, not a raise. At EXECUTION time its per-row counts diverge
    from the Jet field's, so the structural check raises naming the offsets (offsets are data →
    per-partition, exec-time)."""
    record, evt_mask, _jet_mask, muon_mask = _multifield()

    # record-time (2c depth passes: the mask IS ndim-2) → a Plan, NOT a raise
    plan = _write(record, str(tmp_path / "plan"), {0: evt_mask, ("Jet", 1): muon_mask}, compute=False)
    assert isinstance(plan, Plan)

    # execution-time: the Muon offsets diverge from Jet's → GraphedError naming the offsets
    with pytest.raises(GraphedError) as excinfo:
        _write(record, str(tmp_path / "bad"), {0: evt_mask, ("Jet", 1): muon_mask})
    message = str(excinfo.value)
    assert "offsets" in message
    assert "structural" in message


def test_matched_level1_mask_roundtrips_positive_control(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The positive control for the structural refusal: the correctly-shaped per-jet mask (matched Jet
    offsets) writes without raising — so the refusal above is discriminating, not a blanket reject."""
    record, evt_mask, jet_mask, _muon_mask = _multifield()
    paths = _write(record, str(tmp_path / "ok"), {0: evt_mask, ("Jet", 1): jet_mask})
    assert len(paths) == 1


def test_malformed_select_key_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """LOW: a ``select=`` key that is neither a bare depth ``int`` nor a ``(field, depth>=1)`` tuple
    is refused at the call. Positive control: the same record with a valid ``{0: mask}`` writes."""
    record, evt_mask, _jet_mask, _muon_mask = _multifield()
    with pytest.raises(GraphedError) as excinfo:
        _write(record, str(tmp_path / "badkey"), {("Jet",): evt_mask}, compute=False)  # tuple w/o depth
    # discriminates on _normalize_select's OWN early rejection (a downstream check also refuses the
    # key, but with a different message — this pins the trust-boundary guard, not the fallback)
    assert "bare depth" in str(excinfo.value)
    # positive control: a well-formed key on the same record is accepted (returns a Plan)
    assert isinstance(_write(record, str(tmp_path / "ok"), {0: evt_mask}, compute=False), Plan)
