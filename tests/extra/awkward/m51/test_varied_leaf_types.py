"""Every universe of a varied skim reads back from `read_varied` with its in-memory type and values: an
XOR delta stores only same-type `T`/`?T`/`var * T`/`var * ?T` leaves, anything else is stored as-is
under `"base"`."""

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

Build = Callable[[Any], Any]


def _flat(up: Build, nominal: Build = lambda e: e.m) -> Build:
    def build(ev: Any) -> Any:
        v = graphed.vary(nominal(ev), "s", points={"up": up(ev), "same": nominal(ev) * 2})
        return gak.zip({"x": v}, depth_limit=1)

    return build


def _nested(up: Build, nominal: Build = lambda e: e.Jet.pt) -> Build:
    """`Jet.pt` beside a jagged `raw`, so a flat member is broadcast into the Jet lists by the zip."""

    def build(ev: Any) -> Any:
        pt = graphed.vary(nominal(ev), "s", points={"up": up(ev), "same": nominal(ev) * 2})
        return gak.zip({"Jet": gak.zip({"pt": pt, "raw": ev.Jet.pt}), "m": ev.m}, depth_limit=1)

    return build


def _f32(x: Any) -> Any:
    return gak.values_astype(x, "float32")


def _regular(x: Any) -> Any:
    return gak.to_regular(gak.unflatten(x, 1), axis=1)


#: name -> (record builder, stored leaf name, representation of s_up, representation of s_same)
CASES: dict[str, tuple[Build, str, str, str]] = {
    "f64->f32": (_flat(lambda e: _f32(e.m * 1.003)), "x", "base", "xor"),
    "?f64->?f32": (
        _flat(lambda e: _f32(gak.mask(e.m * 1.003, e.m > 0)), lambda e: gak.mask(e.m, e.m > 0)),
        "x",
        "base",
        "xor",
    ),
    "?f64->?f32 with nulls": (
        _flat(lambda e: _f32(gak.mask(e.m * 1.003, e.m > 95)), lambda e: gak.mask(e.m, e.m > 0)),
        "x",
        "base",
        "xor",
    ),
    "i64->f64": (_flat(lambda e: e.n * 1.5, lambda e: e.n), "x", "base", "xor"),
    "f64->i64": (_flat(lambda e: gak.values_astype(e.m, "int64")), "x", "base", "xor"),
    "f64->?f64": (_flat(lambda e: gak.mask(e.m * 3, e.m > 0)), "x", "base", "xor"),
    "?f64->f64": (_flat(lambda e: e.m * 3, lambda e: gak.mask(e.m, e.m > 0)), "x", "base", "xor"),
    "dtype only, equal values": (_flat(lambda e: _f32(e.m)), "x", "base", "xor"),
    "nested jagged Jet.pt f64->f32": (_nested(lambda e: _f32(e.Jet.pt * 1.05)), "Jet_pt", "base", "xor"),
    "jagged->flat": (_flat(lambda e: e.m, lambda e: e.Jet.pt), "x", "base", "xor"),
    "Jet.pt, up a flat f32 broadcast by the zip": (_nested(lambda e: _f32(e.m)), "Jet_pt", "base", "xor"),
    "Jet.pt, nominal a flat broadcast by the zip": (
        _nested(lambda e: e.m * 3, lambda e: e.m),
        "Jet_pt",
        "xor",
        "xor",
    ),
    "?var*f64->?var*f32 with None lists": (
        _flat(lambda e: _f32(gak.mask(e.Jet.pt * 2, e.m > 100)), lambda e: gak.mask(e.Jet.pt, e.m > 100)),
        "x",
        "base",
        "base",
    ),
    "?var*f64, no None lists": (
        _flat(lambda e: gak.mask(e.Jet.pt * 3, e.m > 0), lambda e: gak.mask(e.Jet.pt, e.m > 0)),
        "x",
        "base",
        "base",
    ),
    "regular list": (_flat(lambda e: _regular(e.m * 3), lambda e: _regular(e.m)), "x", "base", "base"),
}

#: same leaf type, different `None` rows: the XOR delta carries the label's own `None`s.
NULL_CASES: dict[str, tuple[Build, str]] = {
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


def _write_and_check(build: Build, tmp_path: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write, read back, assert every universe equals its in-memory member; return the varied manifest
    and awkward's own per-column metadata."""
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    record = build(ev)
    keep = graphed.vary(ev.m > 0, "s", points={"up": ev.m > 95, "same": ev.m > 0})
    paths = ga.to_parquet(record, str(tmp_path / "skim"), select={0: keep})
    assert isinstance(paths, list)
    (path,) = paths
    got = ga.read_varied(path)
    for label in ("nominal", "s_up", "s_same"):
        member: Any = session.materialize(graphed.member_of(record, label))
        expected = member[session.materialize(graphed.member_of(keep, label))]
        disk = got[label]
        assert (str(disk.type), ak.to_list(disk)) == (str(expected.type), ak.to_list(expected)), label
    meta = pq.ParquetFile(path).metadata.metadata
    return json.loads(meta[b"graphed.variations"]), json.loads(meta[b"awkward_array_metadata"])


@pytest.mark.parametrize("case", CASES)
def test_every_universe_reads_back_with_its_own_type_and_values(case: str, tmp_path: Any) -> None:
    build, flat, up_rep, same_rep = CASES[case]
    manifest, _ = _write_and_check(build, tmp_path)
    assert manifest["s_up"][f"__vary_s_up__{flat}"]["representation"] == up_rep
    assert manifest["s_same"][f"__vary_s_same__{flat}"]["representation"] == same_rep


@pytest.mark.parametrize("case", NULL_CASES)
def test_a_universe_keeps_its_own_nones_through_the_xor_delta(case: str, tmp_path: Any) -> None:
    build, flat = NULL_CASES[case]
    manifest, _ = _write_and_check(build, tmp_path)
    assert manifest["s_up"][f"__vary_s_up__{flat}"]["representation"] == "xor"


def test_an_option_delta_without_none_stays_unmasked_on_disk(tmp_path: Any) -> None:
    build, _leaf, _up, _same = CASES["?f64->?f32"]
    _, columns = _write_and_check(build, tmp_path)
    assert columns["__vary_s_same__x"]["mask_type"] == "UnmaskedArray"
