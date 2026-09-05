"""vary-m53 — design §2, §4.10: a plain `vary` over dependent members mints the full grid, all
distinct.

The b-tag SF members depend on the jes-varied jets, so the default is the whole jes(3) x btag(5)
grid — the fifteen universes `EXPECTED_POINTS` names, eight of them machine-minted joints. The
`corpus/m05` `test_variations_produce_distinct_histograms` idiom, extended to that grid: a
resolution that collapsed a joint label onto its one-at-a-time neighbour would show up as a
duplicate fingerprint, which a bare label-set assertion would miss.
"""

from __future__ import annotations

from graphed_corpus.histograms import fingerprint
from m52_corpus_fixtures import EXPECTED_POINTS, joint_program, universe_hist

import graphed


def test_the_auto_fanout_set_is_the_full_grid_and_every_universe_is_distinct() -> None:
    program = joint_program()

    assert graphed.points(program.weight) == EXPECTED_POINTS
    assert sorted(graphed.labels(program.weight)) == sorted(EXPECTED_POINTS)

    fingerprints = {label: fingerprint(universe_hist(program, label)) for label in EXPECTED_POINTS}
    assert len(set(fingerprints.values())) == len(fingerprints), (
        f"universes share a histogram: {fingerprints}"
    )
