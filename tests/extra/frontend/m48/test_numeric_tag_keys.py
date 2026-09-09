"""Numeric tag KEYS: an `int`/`float` where a spelling goes is read as that spelling.

`{+2.5: pt * 1.1, -2.5: pt * 0.9}` mints `jes_25em1` / `jes_m25em1` exactly as the string spellings
do; the collision legs prove the two roads meet in one canonicalization (§1.1).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import graphed
from graphed import Kind, Session
from graphed._tags import canonical_tag, numeric_value
from graphed.context import EventContext
from graphed.errors import GraphedError
from graphed.numpy import NumpyBackend, from_record


def _context() -> tuple[Session, EventContext]:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=np.arange(1.0, 7.0), eta=np.arange(1.0, 7.0) / 6)
    return session, EventContext(
        session, record["pt"], collections={"pt": record["pt"], "eta": record["eta"]}
    )


@pytest.mark.parametrize(
    ("key", "spelling"),
    [
        (2.5, "2.5"),
        (-2.5, "-2.5"),
        (3, "3"),
        (-1, "-1"),
        (2.0, "2"),
        (1e-5, "1e-5"),
        (np.float64(0.1), "0.1"),
        (np.float32(2.5), "2.5"),
        (np.int64(-1), "-1"),
    ],
)
def test_a_numeric_key_is_its_spelling(key: Any, spelling: str) -> None:
    assert canonical_tag(key) == canonical_tag(spelling)
    assert numeric_value(canonical_tag(key)) == Fraction(spelling)


@pytest.mark.parametrize("bad", [True, Fraction(1, 3), float("nan"), float("inf"), 2j, None, b"2"], ids=repr)
def test_a_key_that_is_not_a_tag_is_refused(bad: Any) -> None:
    with pytest.raises(GraphedError):
        canonical_tag(bad)


def test_sigma_keys_mint_ordered_numeric_labels() -> None:
    _session, ctx = _context()
    pt = ctx["pt"]
    varied = graphed.vary(ctx, "jes", collections={"pt": {+2.5: pt * 1.1, -2.5: pt * 0.9}})
    assert graphed.labels(varied) == ("nominal", "jes_25em1", "jes_m25em1")
    assert graphed.variations(varied)["jes"] == {
        "25em1": (Kind.SHIFT, Fraction(5, 2)),
        "m25em1": (Kind.SHIFT, Fraction(-5, 2)),
    }
    # a float coordinate reaches the float-declared family
    corr = varied["pt"] * 0.5
    placed = graphed.vary(
        varied,
        "corr",
        corr,
        is_weight=True,
        points=[("a", corr * 1.3), {"corr": "a", "jes": 2.5}],
    )
    assert "corr_a__jes_25em1" in graphed.labels(placed)


@pytest.mark.parametrize("via", ["collections", "points"])
def test_a_number_and_its_spelling_name_one_universe(via: str) -> None:
    _session, ctx = _context()
    pt = ctx["pt"]
    members = {2.0: pt * 1.1, "2": pt * 0.9}
    with pytest.raises(GraphedError, match="one value cannot name two universes"):
        if via == "collections":
            graphed.vary(ctx, "jes", collections={"pt": members})
        else:
            graphed.vary(pt, "jes", points=members)
