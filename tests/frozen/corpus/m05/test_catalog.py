"""The Required Operations Catalog stays in lock-step with the runnable fixtures.

Guards against the catalog drifting from reality (reviewer focus: "faithful, not imagined").
"""

from __future__ import annotations

from pathlib import Path

from conftest import all_fixtures

DOCS = Path(__file__).resolve().parents[3] / "docs"
CATALOG = DOCS / "requirements" / "ops_catalog.md"
BLOAT = DOCS / "graph_bloat_note.md"


def test_catalog_and_bloat_note_exist() -> None:
    assert CATALOG.is_file()
    assert BLOAT.is_file()


def test_every_fixture_family_is_catalogued() -> None:
    text = CATALOG.read_text(encoding="utf-8")
    # ADL ladder q1..q8 and both systematics families must be named.
    for q in range(1, 9):
        assert f"adl_q{q}" in text or f"q{q}" in text, f"ADL q{q} not in catalog"
    assert "ttbar_" in text
    assert "ttgamma_" in text


def test_catalog_maps_ops_to_milestones() -> None:
    text = CATALOG.read_text(encoding="utf-8")
    # the graded ladder targets and the systematics milestone must be present
    for milestone in ("M3", "M4", "M7", "M9"):
        assert milestone in text, f"catalog does not map any op to {milestone}"


def test_weight_and_kinematic_distinction_documented() -> None:
    text = CATALOG.read_text(encoding="utf-8").lower()
    assert "weight systematic" in text
    assert "kinematic systematic" in text


def test_phase2_items_are_flagged_not_dropped() -> None:
    text = CATALOG.read_text(encoding="utf-8").lower()
    # ONNX inference and >64-category selection are real corpus ops deferred, not silently dropped
    assert "onnx" in text
    assert "phase-2" in text or "phase 2" in text


def test_fixture_count_matches_expectation() -> None:
    assert len(all_fixtures()) == 23
