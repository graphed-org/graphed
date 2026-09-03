"""m51 anchor A — the superset row rule against an INDEPENDENT reference (§6.4a).

When the written rows pass through a varied selection, the writer materializes the SUPERSET: the
rows passing ANY universe's selection (level-0 OR). This anchor pins that against a reference taken
OUTSIDE graphed — per §5.2a a reference derived from the same varied graph would be self-derived and
prove nothing — computed with plain awkward from the same input events:

  * the WRITTEN row set (the union of every universe's reconstructed rows) equals the eager union;
  * each universe's reconstructed rows equal that universe's eager row set;
  * the superset is strictly larger than the smallest universe (so it is a union of universes whose
    row sets genuinely differ) — the JES shift migrates events across the level-0 predicate.

`met_pt` is unvaried and distinct per event, so it is the identity that ties a reconstructed row back
to the input event without asking the graph under test which rows it kept.
"""

from __future__ import annotations

import pytest
from m51_write_fixtures import (
    JES_LABELS,
    as_list,
    eager_superset_met,
    eager_universe_met,
    events_context,
    superset_inputs,
)

import graphed.awkward as ga

pytest.importorskip("pyarrow")


def test_written_superset_is_the_union_and_each_universe_is_its_eager_row_set(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _session, events = events_context()
    record, evt_mask = superset_inputs(events)
    paths = ga.to_parquet(record, str(tmp_path / "sup"), select={0: evt_mask})  # type: ignore[call-arg]
    assert len(paths) == 1
    got = ga.read_varied(paths[0])  # type: ignore[attr-defined]

    assert tuple(got) == JES_LABELS
    # each universe's reconstructed rows == its eager row set (independent reference)
    per_universe: dict[str, list[float]] = {}
    for label in JES_LABELS:
        recon = sorted(as_list(got[label].met_pt))
        per_universe[label] = recon
        assert recon == sorted(as_list(eager_universe_met(label))), label

    # the WRITTEN row set (union of the universes) equals the eager union / superset
    written_union = sorted({v for rows in per_universe.values() for v in rows})
    assert written_union == sorted(set(as_list(eager_superset_met())))

    # and the superset is strictly larger than the smallest universe: a genuine OR over universes
    # whose row sets differ (the JES shift migrates events across the level-0 predicate)
    superset_size = len(set(as_list(eager_superset_met())))
    assert superset_size == len(written_union)
    assert superset_size > min(len(set(rows)) for rows in per_universe.values())
