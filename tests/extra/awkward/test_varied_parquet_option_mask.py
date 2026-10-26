"""A `None` in an option-typed `select=` mask indexes as ak does.

A `None` row (or object) in every universe whose mask holds it, round-tripped through `read_varied`
(graphed#77).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import awkward as ak
import pytest

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

pq = pytest.importorskip("pyarrow.parquet")

DATA = {
    "m": [124.84, 125.5, 130.25, 90.0, 140.0],
    "n": [3, 0, 4, 1, 2],
    "Jet": [[{"pt": 40.5}], [], [{"pt": 55.25}, {"pt": 31.0}], [{"pt": 20.0}], [{"pt": 25.0}]],
}
LABELS = ("nominal", "s_up")


def _record(ev: Any) -> Any:
    v = graphed.vary(ev.m, "s", points={"up": ev.m * 1.01})
    return gak.zip({"x": v, "n": ev.n, "Jet": ev.Jet}, depth_limit=1)


def _named(ev: Any) -> Any:
    return gak.with_name(_record(ev), "Event")


def _event_mask(ev: Any) -> Any:
    return gak.mask(ev.m > 100, ev.n > 0)


def _varied_event_mask(ev: Any) -> Any:
    return graphed.vary(_event_mask(ev), "s", points={"up": gak.mask(ev.m > 125, ev.n > 1)})


def _jet_mask(ev: Any) -> Any:
    return gak.mask(ev.Jet.pt > 30, ev.Jet.pt != 31.0)


def _jet_rows_mask(ev: Any) -> Any:
    return gak.mask(ev.Jet.pt > 30, ev.n != 0)


Select = Callable[[Any], dict[Any, Any]]
CASES: dict[str, tuple[Callable[[Any], Any], Select]] = {
    "level 0": (_record, lambda ev: {0: _event_mask(ev)}),
    "level 0, varied mask": (_record, lambda ev: {0: _varied_event_mask(ev)}),
    "level 0, named record": (_named, lambda ev: {0: _event_mask(ev)}),
    "level 0 option, level 1 plain": (
        _record,
        lambda ev: {0: _event_mask(ev), ("Jet", 1): ev.Jet.pt > 30},
    ),
    "level 1 option objects": (_record, lambda ev: {0: ev.m > 100, ("Jet", 1): _jet_mask(ev)}),
    "level 1 option rows": (_record, lambda ev: {0: ev.m > 100, ("Jet", 1): _jet_rows_mask(ev)}),
}


def _expected(member: ak.Array, masks: dict[Any, ak.Array]) -> ak.Array:
    """Return the analyst's eager selection.

    Rows by the level-0 mask, then each collection's objects.
    """
    m0 = masks.get(0)
    out = member if m0 is None else member[m0]
    for key, mask in masks.items():
        if key != 0:
            field = key[0]
            out = ak.with_field(out, out[field][mask if m0 is None else mask[m0]], field)
    return out


@pytest.mark.parametrize("case", CASES)
def test_an_option_mask_selects_as_ak_indexing_does(tmp_path: Any, case: str) -> None:
    build, select = CASES[case]
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    record, masks = build(ev), select(ev)
    paths = ga.to_parquet(record, str(tmp_path / "skim"), select=masks)
    assert isinstance(paths, list)
    (path,) = paths
    got = ga.read_varied(path)
    assert list(got) == list(LABELS)
    for label in LABELS:
        member = session.materialize(graphed.member_of(record, label))
        eager = {k: session.materialize(graphed.member_of(m, label)) for k, m in masks.items()}
        want = _expected(member, eager)
        assert ak.to_list(got[label]) == ak.to_list(want), label
        assert str(got[label].type) == str(want.type), label
        assert "None" in str(ak.to_list(want)), "the case has no None to carry"


def test_an_option_mask_is_stored_as_is_and_a_plain_one_as_packbits(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    paths = ga.to_parquet(
        _record(ev),
        str(tmp_path / "skim"),
        select={0: _event_mask(ev), ("Jet", 1): ev.Jet.pt > 30},
    )
    assert isinstance(paths, list)
    (path,) = paths
    manifest = json.loads(pq.ParquetFile(path).metadata.metadata[b"graphed.variations"])
    reps = {
        col: info["representation"]
        for col, info in manifest["nominal"].items()
        if "__mask__" in col
    }
    assert reps == {"__vary_nominal__mask__0": "option", "__vary_nominal__mask__Jet_1": "packbits"}
    assert len(ak.from_parquet(path)) == 4  # the row whose mask is None in every universe is kept
