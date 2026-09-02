"""§5.3: the projection union a shift grows, and the per-label read-width stats beside it."""

from __future__ import annotations

from m49_vary_fixtures import flat_projection_program

import graphed
from graphed import read_columns

EXTRA_COLUMN = "Jet_eta"


def test_a_shift_grows_the_union_by_exactly_its_extra_column() -> None:
    """The plain-union half rides its own OUTPUT SET: in the full varied program the conservative
    member collapses the union to `None` (§2.3d) and this assertion would go vacuous."""
    program = flat_projection_program()

    assert read_columns([program.nominal], program.source_nid) == ("Jet_pt",)
    assert read_columns([program.nominal, program.shifted], program.source_nid) == ("Jet_eta", "Jet_pt")


def test_one_conservative_member_collapses_the_plain_union() -> None:
    program = flat_projection_program()
    assert read_columns([program.varied], program.source_nid) is None


def test_stats_report_the_shifted_labels_extra_column() -> None:
    program = flat_projection_program()
    stats = graphed.read_columns_by_label([program.varied], program.source_nid)

    assert tuple(stats) == graphed.labels(program.varied)
    assert set(stats["jes_up"]) - set(stats["nominal"]) == {EXTRA_COLUMN}
    assert set(stats["nominal"]) - set(stats["jes_up"]) == set()
    assert stats["nominal"] == tuple(sorted(stats["nominal"]))
    assert stats["jes_up"] == tuple(sorted(stats["jes_up"]))


def test_a_whole_record_consumer_reports_none_not_empty() -> None:
    """`None` means "read every column"; `()` would say the opposite, so the two must not merge."""
    program = flat_projection_program()
    stats = graphed.read_columns_by_label([program.varied], program.source_nid)

    assert stats["jes_opaque"] is None
    assert stats["nominal"] is not None


def test_the_mapping_form_answers_the_same_stats() -> None:
    program = flat_projection_program()
    container = program.varied
    mapping = {label: [graphed.universe(container, label)] for label in graphed.labels(container)}

    expected = graphed.read_columns_by_label([container], program.source_nid)
    assert graphed.read_columns_by_label(mapping, program.source_nid) == expected
