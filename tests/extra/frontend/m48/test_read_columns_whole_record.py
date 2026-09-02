"""F7: an External handed the source RECORD reads every column, inside one graph too.

The m48 change made `read_columns` `None`-dominant ACROSS graphs. The same masking survived
WITHIN one graph, because the walk's external callback never set the conservative flag: a graph
whose External consumes the whole record and which separately reads a field narrowed to that
field, and `aggregate_plan` — the sole `src` consumer — routes straight into it.
"""

from __future__ import annotations

import numpy as np

import graphed
from graphed import Session, read_columns
from graphed.numpy import NumpyBackend, from_record


def _source() -> tuple[Session, graphed.Array]:
    session = Session(NumpyBackend())
    return session, from_record(session, "r", pt=np.arange(1.0, 5.0), eta=np.arange(4.0))


def test_an_external_over_the_whole_record_makes_one_graph_conservative() -> None:
    _s, record = _source()
    mixed = graphed.apply(lambda rec, pt: pt, record, record["pt"], name="uses_whole_record")
    assert read_columns([mixed], record.node_id) is None


def test_an_external_over_FIELDS_alone_still_narrows() -> None:
    """The control that keeps the fix from degenerating into "any External reads everything":
    projection must still flow through an External to its declared field inputs."""
    _s, record = _source()
    fields_only = graphed.apply(lambda pt, eta: pt, record["pt"], record["eta"], name="fields_only")
    assert read_columns([fields_only], record.node_id) == ("eta", "pt")
    assert read_columns([record["pt"] * 2.0], record.node_id) == ("pt",)


def test_the_whole_record_read_dominates_a_narrowing_sibling_in_the_same_graph() -> None:
    """The masking this repairs: `needed` was non-empty, so the graph reported that narrow set
    while an External beside it needed every column."""
    _s, record = _source()
    mixed = graphed.apply(lambda rec, pt: pt, record, record["pt"], name="uses_whole_record")
    combined = mixed * record["eta"]
    assert read_columns([combined], record.node_id) is None
