"""Shared fixtures for the M0.5 frozen suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import awkward as ak
import pytest
from hist import Hist

from graphed_corpus import (
    ADL_QUERIES,
    TTBAR_FIXTURES,
    TTGAMMA_FIXTURES,
    make_events,
    ttbar_region,
    ttgamma_region,
)

REF_DIR = Path(__file__).resolve().parents[3] / "corpus" / "references"


def all_fixtures() -> dict[str, Callable[[ak.Array], Hist]]:
    """Map every canonical-analysis fixture name -> a callable(events) -> Hist."""
    reg: dict[str, Callable[[ak.Array], Hist]] = {}
    for q, fn in ADL_QUERIES.items():
        reg[f"adl_{q}"] = fn
    for name, (region, var) in TTBAR_FIXTURES.items():
        reg[name] = lambda ev, region=region, var=var: ttbar_region(ev, region=region, variation=var)
    for name, var in TTGAMMA_FIXTURES.items():
        reg[name] = lambda ev, var=var: ttgamma_region(ev, variation=var)
    return reg


@pytest.fixture(scope="session")
def events() -> ak.Array:
    return make_events()


@pytest.fixture(scope="session")
def registry() -> dict[str, Callable[[ak.Array], Hist]]:
    return all_fixtures()
