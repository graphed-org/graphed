"""A universe whose leaf type differs from nominal's reads back from `read_varied` with its own type and
values (GW2): an XOR delta needs one dtype, so such a leaf is stored as-is under `"base"`."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import awkward as ak
import pyarrow.parquet as pq
import pytest

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

DATA = {
    "m": [124.84, 125.5, 130.25, 90.0],
    "n": [3, 1, 4, 1],
    "Jet": [[{"pt": 40.5}], [], [{"pt": 55.25}, {"pt": 31.0}], [{"pt": 20.0}]],
}


def _flat(up: Callable[[Any], Any], nominal: Callable[[Any], Any] = lambda e: e.m) -> Callable[[Any], Any]:
    def build(ev: Any) -> Any:
        return gak.zip({"x": graphed.vary(nominal(ev), "s", points={"up": up(ev), "same": nominal(ev) * 2})})

    return build


def _nested(
    up: Callable[[Any], Any], nominal: Callable[[Any], Any] = lambda e: e.Jet.pt
) -> Callable[[Any], Any]:
    def build(ev: Any) -> Any:
        pt = graphed.vary(nominal(ev), "s", points={"up": up(ev), "same": nominal(ev) * 2})
        return gak.zip({"Jet": gak.zip({"pt": pt}), "m": ev.m}, depth_limit=1)

    return build


def _f32(x: Any) -> Any:
    return gak.values_astype(x, "float32")


CASES: dict[str, tuple[Callable[[Any], Any], str]] = {
    "f64->f32": (_flat(lambda e: _f32(e.m * 1.003)), "x"),
    "?f64->?f32": (
        _flat(lambda e: _f32(gak.mask(e.m * 1.003, e.m > 0)), lambda e: gak.mask(e.m, e.m > 0)),
        "x",
    ),
    "?f64->?f32 with nulls": (
        _flat(lambda e: _f32(gak.mask(e.m * 1.003, e.m > 95)), lambda e: gak.mask(e.m, e.m > 0)),
        "x",
    ),
    "i64->f64": (_flat(lambda e: e.n * 1.5, lambda e: e.n), "x"),
    "f64->i64": (_flat(lambda e: gak.values_astype(e.m, "int64")), "x"),
    "f64->?f64": (_flat(lambda e: gak.mask(e.m * 3, e.m > 0)), "x"),
    "?f64->f64": (_flat(lambda e: e.m * 3, lambda e: gak.mask(e.m, e.m > 0)), "x"),
    "dtype only, equal values": (_flat(lambda e: _f32(e.m)), "x"),
    "nested jagged Jet.pt f64->f32": (_nested(lambda e: _f32(e.Jet.pt * 1.05)), "Jet_pt"),
}

#: same leaf type, different `None` rows: the XOR delta carries the label's own `None`s.
NULL_CASES: dict[str, tuple[Callable[[Any], Any], str]] = {
    "flat, None in nominal only": (
        _flat(lambda e: gak.mask(e.m * 2, e.m > 0), lambda e: gak.mask(e.m, e.m > 125)),
        "x",
    ),
    "flat, None in up only": (
        _flat(lambda e: gak.mask(e.m * 2, e.m > 125), lambda e: gak.mask(e.m, e.m > 0)),
        "x",
    ),
    "jagged, None in nominal only": (
        _nested(lambda e: gak.mask(e.Jet.pt * 2, e.Jet.pt > 0), lambda e: gak.mask(e.Jet.pt, e.Jet.pt > 35)),
        "Jet_pt",
    ),
}


def _write_and_read(
    build: Callable[[Any], Any], tmp_path: Any
) -> tuple[Session, Any, Any, dict[str, Any], Any]:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    record = build(ev)
    keep = graphed.vary(ev.m > 0, "s", points={"up": ev.m > 95, "same": ev.m > 0})
    paths = ga.to_parquet(record, str(tmp_path / "skim"), select={0: keep})
    assert isinstance(paths, list)
    (path,) = paths
    manifest = json.loads(pq.ParquetFile(path).metadata.metadata[b"graphed.variations"])
    return session, record, keep, manifest, ga.read_varied(path)


def _assert_universes_read_back(build: Callable[[Any], Any], tmp_path: Any) -> dict[str, Any]:
    session, record, keep, manifest, got = _write_and_read(build, tmp_path)
    for label in ("nominal", "s_up", "s_same"):
        member: Any = session.materialize(graphed.member_of(record, label))
        expected = member[session.materialize(graphed.member_of(keep, label))]
        disk = got[label]
        assert (str(disk.type), ak.to_list(disk)) == (str(expected.type), ak.to_list(expected)), label
    return manifest


@pytest.mark.parametrize("case", CASES)
def test_every_universe_reads_back_with_its_own_type_and_values(case: str, tmp_path: Any) -> None:
    build, flat = CASES[case]
    manifest = _assert_universes_read_back(build, tmp_path)
    # only the type-changing universe leaves the XOR representation
    assert manifest["s_up"][f"__vary_s_up__{flat}"]["representation"] == "base"
    assert manifest["s_same"][f"__vary_s_same__{flat}"]["representation"] == "xor"


@pytest.mark.parametrize("case", NULL_CASES)
def test_a_universe_keeps_its_own_nones_through_the_xor_delta(case: str, tmp_path: Any) -> None:
    build, flat = NULL_CASES[case]
    manifest = _assert_universes_read_back(build, tmp_path)
    assert manifest["s_up"][f"__vary_s_up__{flat}"]["representation"] == "xor"
