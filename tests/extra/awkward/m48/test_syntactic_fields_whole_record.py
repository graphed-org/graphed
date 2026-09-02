"""C2: `io._syntactic_fields` is the structural twin of `projection.read_columns`'s walk.

It feeds `_evaluation_columns`, i.e. the per-task parquet read list, so the same masking has a
harder consequence there: evaluation replays every recorded node, and a field the read list omits
is a field the External replays against and cannot find.
"""

from __future__ import annotations

from typing import Any

import awkward as ak

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.awkward.io import _syntactic_fields

EVENTS = ak.Array([{"pt": 1.0, "eta": 0.5, "phi": 0.1}, {"pt": 2.0, "eta": 1.5, "phi": 0.2}])


def _source() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", EVENTS)


def test_an_external_over_the_whole_record_makes_the_read_list_conservative() -> None:
    _s, root = _source()
    mixed = graphed.apply(lambda rec, pt: pt, root, root.pt, name="uses_whole_record")
    assert _syntactic_fields(mixed, root.node_id) is None


def test_an_external_reaching_a_field_THROUGH_the_record_is_not_narrowed_away() -> None:
    """The sharper case: the External replays against `phi`, which no field op records, so a read
    list of {'pt'} would hand evaluation a chunk without the column it needs."""
    _s, root = _source()
    scored = graphed.apply(lambda rec: rec.phi, root, name="reads_phi_internally")
    graph = graphed.apply(lambda score, pt: pt, scored, root.pt, name="combine")
    assert _syntactic_fields(graph, root.node_id) is None


def test_an_external_over_FIELDS_alone_still_narrows() -> None:
    """The control that keeps the repair from degenerating into "every External reads everything"."""
    _s, root = _source()
    fields_only = graphed.apply(lambda pt, eta: pt, root.pt, root.eta, name="fields_only")
    assert _syntactic_fields(fields_only, root.node_id) == {"eta", "pt"}
    assert _syntactic_fields(root.pt * 2.0, root.node_id) == {"pt"}
