"""m51 anchor D — the "record is pre-selection" entry checks (§6.4a), each with the control it decides.

The decidable entry check splits by SITE (trap ledger §7), and this suite freezes the split so an
implementation cannot move a predicate to the wrong site:

  RECORD-TIME (raised at the `to_parquet` call, so `compute=False` still raises):
    (2a) lineage · (2c) depth at every supplied level · bare-key ambiguity · §6.4b row-space.
  EXECUTION-TIME (raised per partition from `_WritePart` before any buffer is stored, so
    `compute=False` returns a `Plan` and RUNNING it raises):
    (1) multiplicity offsets · (2b) row-count · the level-≥1 structural half.

Each predicate is pinned with the POSITIVE control it decides, so a diagnostic that blames the wrong
property is caught (the §6.1d loose-value / §6.4b row-space-not-offsets standard).
"""

from __future__ import annotations

from typing import Any

import pytest
from m51_write_fixtures import (
    awkward_session,
    events_context,
    multifield_skim_inputs,
    multiplicity_change_inputs,
    rowspace_negative_inputs,
    rowspace_positive_inputs,
)

import graphed
import graphed.awkward as ga
from graphed.awkward import gak
from graphed.core.execution import Plan
from graphed.errors import GraphedError

pytest.importorskip("pyarrow")


def _write(record: Any, dest: str, select: Any, *, compute: bool = True) -> Any:
    return ga.to_parquet(record, dest, select=select, compute=compute)  # type: ignore[call-arg]


# ---- (1) MULTIPLICITY — EXECUTION-time offsets refusal ---------------------------------------
def test_multiplicity_change_is_refused_at_execution_time_not_record_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = multiplicity_change_inputs(events)

    # record-time must NOT raise (row/offset shapes are DATA, §6.4a): compute=False returns a Plan
    plan = _write(record, str(tmp_path / "mult"), {0: evt_mask}, compute=False)
    assert isinstance(plan, Plan)
    # running it surfaces the offsets refusal through the executor's error path
    with pytest.raises(GraphedError):
        _write(record, str(tmp_path / "mult2"), {0: evt_mask})


# ---- (2a) LINEAGE — record-time -------------------------------------------------------------
def test_2a_refuses_a_record_whose_context_the_mask_does_not_derive_from(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Silent-corruption case: a record read from a NON-varied embedded selection, written with a
    varied mask over a sibling (jes) context, is refused at the CALL naming both contexts."""
    _s, events = events_context()
    nominal_mask = gak.num(events.Jet) >= 2
    sel = events[nominal_mask]  # a non-varied embedded selection
    jes = _jes(events)
    varied_mask = gak.any(jes.Jet.pt > 30.0, axis=1)  # a Varied mask over the sibling jes context
    with pytest.raises(GraphedError):
        _write(sel.Jet, str(tmp_path / "corrupt"), varied_mask, compute=False)


def test_2a_refuses_a_chained_context_mask_against_a_root_row_space_record(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    sel = events[gak.num(events.Jet) >= 2]
    sel2 = sel[gak.num(sel.Muon) >= 1]  # one row space below `sel`
    with pytest.raises(GraphedError):
        _write(events.Jet, str(tmp_path / "chain"), graphed.selection(sel2), compute=False)  # type: ignore[attr-defined]


def test_2a_is_skipped_for_a_context_free_record(tmp_path) -> None:
    """Absent-operand (i): a loose §2.1a record carries no handle, so (2a) is skipped and the write
    proceeds — the positive control that keeps the loose sink reachable."""
    _session, root = awkward_session()
    assert graphed.context_of(root) is None
    mask = gak.num(root.Jet) >= 2  # context-free too
    paths = _write(root.Jet, str(tmp_path / "loose"), mask)
    got = ga.read_varied(paths[0])  # type: ignore[attr-defined]
    assert list(got) == ["nominal"]


def test_2a_refuses_a_contexted_record_with_a_handleless_mask(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Absent-operand (ii): a contexted record and a mask that carries NO handle (§2.3e's Drop rule)
    is refused — worded over the handle, not "derived no context"."""
    _session, root = awkward_session()
    events = ga.gnano.events(root)
    handleless = gak.num(root.Jet) >= 2  # read through the raw root, so it carries no context handle
    assert graphed.context_of(events.Jet) is events and graphed.context_of(handleless) is None
    with pytest.raises(GraphedError):
        _write(events.Jet, str(tmp_path / "nohandle"), handleless, compute=False)


def test_2a_accepts_a_mask_that_originates_the_records_handle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The §2.3e ORIGINATION control that discriminates "carries no handle" from "derived no
    context": a mask recorded entirely through the record's own context carries that handle and is
    ACCEPTED, even though no context was ever derived from it."""
    _s, events = events_context()
    mask = events.MET.pt > 0.0  # read THROUGH the context -> carries `events` by origination
    assert graphed.context_of(mask) is events
    paths = _write(events.Jet, str(tmp_path / "orig"), mask)
    assert len(paths) == 1


# ---- (2c) DEPTH — record-time, at every supplied level --------------------------------------
def test_2c_refuses_a_jagged_level_0_mask_that_passes_2a_and_2b(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The discriminator is DEPTH ALONE: a per-jet (jagged) mask read through the record's own
    context passes (2a) lineage and (2b) row-count, but a flat level-0 mask over the SAME context
    is accepted — so only depth separates refusal from acceptance."""
    _s, events = events_context()
    jagged = events.Jet.pt > 25.0  # ndim 2 — per object
    flat = gak.num(events.Jet) >= 2  # ndim 1 — per event, same context
    with pytest.raises(GraphedError):
        _write(events.Jet, str(tmp_path / "jag"), jagged, compute=False)
    assert len(_write(events.Jet, str(tmp_path / "flat"), flat)) == 1  # depth is the only difference


def test_2c_refuses_a_too_shallow_mask_supplied_at_level_1(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The mirror at level ≥ 1: a FLAT mask supplied at level 1 is refused at the call naming the
    level (otherwise the level-≥1 structural check has no operand and dies per partition)."""
    _s, events = events_context()
    record, evt_mask, jet_mask = multifield_skim_inputs(events)
    with pytest.raises(GraphedError):
        _write(record, str(tmp_path / "shallow"), {0: evt_mask, ("Jet", 1): evt_mask}, compute=False)
    # positive control: the correctly-shaped per-jet mask at level 1 is a valid record-time form
    assert isinstance(
        _write(record, str(tmp_path / "ok"), {0: evt_mask, ("Jet", 1): jet_mask}, compute=False), Plan
    )


# ---- bare-key ambiguity — record-time -------------------------------------------------------
def test_bare_depth_key_on_two_independently_jagged_fields_is_ambiguous(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§6.4a: a record with two independently jagged depth-1 fields refuses a BARE `1` key naming the
    field paths; the field-scoped `("Jet", 1)` spelling on the SAME record is the positive control."""
    _s, events = events_context()
    record = gak.zip({"Jet": events.Jet, "Muon": events.Muon}, depth_limit=1)
    evt_mask = gak.num(events.Jet) >= 2
    jet_mask = events.Jet.pt > 25.0
    with pytest.raises(GraphedError):
        _write(record, str(tmp_path / "ambig"), {0: evt_mask, 1: jet_mask}, compute=False)
    assert isinstance(
        _write(record, str(tmp_path / "scoped"), {0: evt_mask, ("Jet", 1): jet_mask}, compute=False), Plan
    )


# ---- §6.4b ROW-SPACE — record-time, blamed as row space not offsets --------------------------
def test_6_4b_refuses_a_selection_scoped_stored_weight_naming_the_row_space(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = rowspace_negative_inputs(events)
    with pytest.raises(GraphedError) as excinfo:
        _write(record, str(tmp_path / "rs"), {0: evt_mask}, compute=False)
    message = str(excinfo.value).lower()
    assert "row" in message and "space" in message  # blames the row space...
    assert "offset" not in message  # ...NOT the offsets (§6.1d/§6.4b message-split standard)


def test_6_4b_accepts_a_vary_reached_stored_weight(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = rowspace_positive_inputs(events)
    assert len(_write(record, str(tmp_path / "rs_ok"), {0: evt_mask})) == 1


# ---- shared helper --------------------------------------------------------------------------
def _jes(events: Any) -> Any:
    jets = events.Jet
    up = gak.with_field(jets, jets.pt * 1.05, "pt")
    down = gak.with_field(jets, jets.pt * 0.95, "pt")
    return graphed.vary(events, "jes", collections={"Jet": {"up": up, "down": down}})
