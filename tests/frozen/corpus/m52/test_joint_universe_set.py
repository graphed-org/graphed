"""vary-m52-C6 — design §4.4 / §4.10: the registered set is the symmetric four, all distinct.

The `corpus/m05` `test_variations_produce_distinct_histograms` idiom, extended to the joint set: a
resolution that collapses a joint label onto its one-at-a-time neighbour shows up as a duplicate
fingerprint, which is what a bare label-set assertion would miss.
"""

from __future__ import annotations

from graphed_corpus.histograms import fingerprint
from m52_corpus_fixtures import EXPECTED_POINTS, joint_program, universe_hist

import graphed


def test_the_joint_set_is_the_symmetric_four_and_every_universe_is_distinct() -> None:
    program = joint_program()

    assert graphed.points(program.weight) == EXPECTED_POINTS
    assert sorted(graphed.labels(program.weight)) == sorted(EXPECTED_POINTS)

    fingerprints = {label: fingerprint(universe_hist(program, label)) for label in EXPECTED_POINTS}
    assert len(set(fingerprints.values())) == len(fingerprints), (
        f"universes share a histogram: {fingerprints}"
    )
