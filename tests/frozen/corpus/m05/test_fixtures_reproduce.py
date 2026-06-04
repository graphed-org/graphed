"""Every canonical analysis runs and reproduces its stored reference bit-for-bit.

This is the core M0.5 deliverable: the reference outputs later milestones assert ``graphed``
matches. A missing reference or a mismatch is a failure, not a silent gap.
"""

from __future__ import annotations

import awkward as ak
import pytest
from conftest import REF_DIR, all_fixtures
from hist import Hist

from graphed_corpus import fingerprint, load_reference
from graphed_corpus.histograms import bin_values

FIXTURE_NAMES = sorted(all_fixtures())


def test_expected_fixture_count() -> None:
    # 8 ADL + (2 regions x 5 variations) ttbar + 5 ttgamma = 23
    assert len(FIXTURE_NAMES) == 23


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_reference_exists(name: str) -> None:
    assert (REF_DIR / f"{name}.json").exists(), f"missing reference for {name}"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_reproduces_reference(name: str, events: ak.Array, registry: dict[str, object]) -> None:
    fn = all_fixtures()[name]
    h: Hist = fn(events)
    ref = load_reference(REF_DIR / f"{name}.json")
    assert bin_values(h) == ref["values"], f"{name}: bin contents drifted from reference"
    assert fingerprint(h) == ref["fingerprint"], f"{name}: fingerprint drifted from reference"
