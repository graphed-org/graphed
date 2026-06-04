"""Shared fixtures for the M3 frozen suite."""

from __future__ import annotations

import awkward as ak
import pytest
from graphed_corpus import make_events


@pytest.fixture(scope="session")
def shared_events() -> ak.Array:
    return make_events(n_events=2000, seed=1234)
