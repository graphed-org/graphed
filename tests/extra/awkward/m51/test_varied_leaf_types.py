"""Every universe of a varied skim reads back from `read_varied` with its in-memory type and values: an
XOR delta stores only same-type `T`/`?T`/`var * T`/`var * ?T` leaves, anything else is stored as-is
under `"base"`."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.awkward as ga
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

pq = pytest.importorskip("pyarrow.parquet")

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


def _f16(x: Any) -> Any:
    return gak.values_astype(x, "float16")


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


def _write_and_check(
    build: Build, tmp_path: Any, *, exact_type: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        assert ak.to_list(disk) == ak.to_list(expected), label
        assert not exact_type or str(disk.type) == str(expected.type), label
    meta = pq.ParquetFile(path).metadata.metadata
    return json.loads(meta[b"graphed.variations"]), json.loads(meta[b"awkward_array_metadata"])


@pytest.mark.parametrize("case", CASES)
def test_every_universe_reads_back_with_its_own_type_and_values(case: str, tmp_path: Any) -> None:
    build, flat, up_rep, same_rep = CASES[case]
    manifest, _ = _write_and_check(build, tmp_path)
    for label, rep in (("s_up", up_rep), ("s_same", same_rep)):
        assert manifest[label][f"__vary_{label}__{flat}"]["representation"] == rep
        # a "base" label also carries an xor marker naming no leaf, which a pre-"base" reader refuses
        marker = manifest[label].get(f"requires_base__{label}__{flat}")
        assert (marker or {}).get("representation") == ("xor" if rep == "base" else None), label


@pytest.mark.parametrize("case", NULL_CASES)
def test_a_universe_keeps_its_own_nones_through_the_xor_delta(case: str, tmp_path: Any) -> None:
    build, flat = NULL_CASES[case]
    manifest, _ = _write_and_check(build, tmp_path)
    assert manifest["s_up"][f"__vary_s_up__{flat}"]["representation"] == "xor"


def test_an_option_delta_without_none_stays_unmasked_on_disk(tmp_path: Any) -> None:
    build, _leaf, _up, _same = CASES["?f64->?f32"]
    _, columns = _write_and_check(build, tmp_path)
    assert columns["__vary_s_same__x"]["mask_type"] == "UnmaskedArray"


def test_a_datetime64_s_leaf_is_stored_as_is(tmp_path: Any) -> None:
    def dt(x: Any) -> Any:
        return gak.values_astype(x, "datetime64[s]")

    def build(ev: Any) -> Any:
        v = graphed.vary(dt(ev.n), "s", points={"up": dt(ev.n * 2), "same": dt(ev.n * 3)})
        return gak.zip({"x": v}, depth_limit=1)

    # parquet itself stores datetime64[s] as [ms], for nominal too, so only the values are compared
    manifest, _ = _write_and_check(build, tmp_path, exact_type=False)
    assert manifest["s_up"]["__vary_s_up__x"]["representation"] == "base"


@pytest.mark.parametrize(
    ("ups", "reps"),
    [
        ({"y__requires_base": lambda x: x * 2}, {"__vary_s_up__y__requires_base": "xor"}),
        (
            {"x": lambda x: _f32(x * 2), "x__requires_base": lambda x: x * 2},
            {
                "__vary_s_up__x": "base",
                "__vary_s_up__x__requires_base": "xor",
                "requires_base__s_up__x": "xor",
                "type__s_up": "base",
            },
        ),
    ],
)
def test_a_leaf_named_like_the_marker_is_an_ordinary_column(
    ups: dict[str, Build], reps: dict[str, str], tmp_path: Any
) -> None:
    def build(ev: Any) -> Any:
        varied = {
            name: graphed.vary(ev.m, "s", points={"up": up(ev.m), "same": ev.m * 3})
            for name, up in ups.items()
        }
        return gak.zip(varied, depth_limit=1)

    manifest, _ = _write_and_check(build, tmp_path)
    assert {
        col: info["representation"] for col, info in manifest["s_up"].items() if "mask" not in col
    } == reps


def _doc(x: Any, doc: str) -> Any:
    return gak.with_parameter(x, "__doc__", doc)


def _jet_record(pt: Any, ev: Any) -> Any:
    return gak.zip({"Jet": gak.zip({"pt": pt}), "m": ev.m}, depth_limit=1)


#: name -> (record builder, whether s_up's record type differs from nominal's in a parameter)
PARAM_CASES: dict[str, tuple[Build, bool]] = {
    "flat, same __doc__": (_flat(lambda e: _doc(e.m * 3, "mass"), lambda e: _doc(e.m, "mass")), False),
    "?flat, same __doc__": (
        _flat(lambda e: _doc(gak.mask(e.m * 3, e.m > 0), "m"), lambda e: _doc(gak.mask(e.m, e.m > 0), "m")),
        False,
    ),
    "jagged list, same __doc__": (
        _flat(lambda e: _doc(e.Jet.pt * 3, "pt"), lambda e: _doc(e.Jet.pt, "pt")),
        False,
    ),
    "__doc__ on up only": (_flat(lambda e: _doc(e.m * 3, "up")), True),
    "__doc__ differs": (_flat(lambda e: _doc(e.m * 3, "b"), lambda e: _doc(e.m, "a")), True),
    "float16 up, float64 nominal": (_flat(lambda e: gak.values_astype(e.m * 3, "float16")), True),
    "float16 nominal, __doc__ on up": (
        _flat(lambda e: _doc(_f16(e.m * 3), "u"), lambda e: _f16(e.m)),
        True,
    ),
    "Jet list __doc__ lifted by zip": (
        lambda e: _jet_record(
            graphed.vary(
                _doc(e.Jet.pt, "pt"),
                "s",
                points={"up": _doc(e.Jet.pt * 3, "pt"), "same": _doc(e.Jet.pt * 2, "pt")},
            ),
            e,
        ),
        False,
    ),
    "named record, shifted collection": (
        lambda e: gak.zip(
            {
                "Jet": graphed.vary(
                    gak.with_name(e.Jet, "Momentum"),
                    "s",
                    points={
                        "up": gak.with_field(gak.with_name(e.Jet, "Momentum"), e.Jet.pt * 3, "pt"),
                        "same": gak.with_field(gak.with_name(e.Jet, "Momentum"), e.Jet.pt * 2, "pt"),
                    },
                ),
                "m": e.m,
            },
            depth_limit=1,
        ),
        False,
    ),
}


@pytest.mark.parametrize("case", PARAM_CASES)
def test_every_universe_reads_back_with_its_parameters(case: str, tmp_path: Any) -> None:
    build, stored = PARAM_CASES[case]
    manifest, _ = _write_and_check(build, tmp_path)
    # a universe whose type matches nominal's, parameters included, adds nothing to the file
    assert ("type__s_up" in manifest["s_up"]) is stored
    assert "type__s_same" not in manifest["s_same"]
    assert "type__nominal" not in manifest["nominal"]


@pytest.mark.parametrize(
    "top",
    [lambda r: gak.with_name(r, "Event"), lambda r: gak.with_parameter(r, "__doc__", "event")],
    ids=["with_name", "__doc__"],
)
@pytest.mark.parametrize("with_jet", [False, True], ids=["flat", "with Jet"])
def test_every_universe_keeps_the_top_records_parameters(
    top: Callable[[Any], Any], with_jet: bool, tmp_path: Any
) -> None:
    def build(ev: Any) -> Any:
        x = graphed.vary(ev.m, "s", points={"up": ev.m * 3, "same": ev.m * 2})
        return top(gak.zip({"x": x, "Jet": ev.Jet} if with_jet else {"x": x}, depth_limit=1))

    manifest, _ = _write_and_check(build, tmp_path)
    assert "type__nominal" in manifest["nominal"]
    assert "type__s_up" not in manifest["s_up"]


def test_a_varied_field_set_on_the_source_record_reads_back(tmp_path: Any) -> None:
    def build(ev: Any) -> Any:
        return gak.with_field(ev, graphed.vary(ev.m, "s", points={"up": ev.m * 3, "same": ev.m * 2}), "m")

    manifest, _ = _write_and_check(build, tmp_path)
    assert manifest["s_up"]["__vary_s_up__m"]["representation"] == "xor"


def test_a_file_without_the_record_type_entry_keeps_each_universes_own_type(tmp_path: Any) -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    record = _flat(lambda e: _f32(e.m * 3))(ev)
    paths = ga.to_parquet(record, str(tmp_path / "skim"), select={0: ev.m > 0})
    assert isinstance(paths, list)
    table = pq.read_table(paths[0])
    meta = dict(table.schema.metadata)
    manifest = json.loads(meta[b"graphed.variations"])
    del manifest["s_up"]["type__s_up"]  # as the writer did before the entry existed
    meta[b"graphed.variations"] = json.dumps(manifest, sort_keys=True).encode()
    old = str(tmp_path / "old.parquet")
    pq.write_table(table.replace_schema_metadata(meta), old)
    member: Any = session.materialize(graphed.member_of(record, "s_up"))
    got = ga.read_varied(old)["s_up"]
    assert (str(got.type), ak.to_list(got)) == (str(member.type), ak.to_list(member))


class _CountingStore:
    def __init__(self, store: Any) -> None:
        self.store, self.calls = store, 0

    def nodes(self) -> Any:
        self.calls += 1
        return self.store.nodes()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)


def _store_snapshots(n_labels: int, tmp_path: Any) -> int:
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "ev", ak.Array(DATA))
    points = {f"p{i}": ev.m * (i + 2) for i in range(n_labels)}
    record = gak.with_name(gak.with_field(ev, graphed.vary(ev.m, "s", points=points), "m"), "Event")
    counting = _CountingStore(session._store)
    session._store = counting  # type: ignore[assignment]
    ga.to_parquet(record, str(tmp_path / f"n{n_labels}"), select={0: ev.m > 0}, compute=False)
    return counting.calls


def test_resolving_varied_leaves_snapshots_the_store_independently_of_the_label_count(tmp_path: Any) -> None:
    assert _store_snapshots(2, tmp_path) == _store_snapshots(8, tmp_path)


@pytest.mark.parametrize(
    "top",
    [lambda r: gak.with_name(r, "Event"), lambda r: gak.with_parameter(r, "__doc__", "event")],
    ids=["with_name", "with_parameter"],
)
def test_a_named_record_stores_deltas_only_for_its_varied_leaves(
    top: Callable[[Any], Any], tmp_path: Any
) -> None:
    def build(ev: Any) -> Any:
        x = graphed.vary(ev.m, "s", points={"up": ev.m * 3, "same": ev.m * 2})
        return top(gak.zip({"x": x, "n": ev.n, "Jet": ev.Jet}, depth_limit=1))

    manifest, _ = _write_and_check(build, tmp_path)
    values = [col for col in manifest["s_up"] if col.startswith("__vary_s_up__") and "__mask__" not in col]
    assert values == ["__vary_s_up__x"]


@pytest.mark.parametrize("varied_y", [False, True], ids=["x", "x_y"])
@pytest.mark.parametrize("scalar", [1.5, np.float32(1.5)], ids=repr)
def test_a_scalar_field_equal_in_every_universe_stores_no_delta(
    scalar: Any, varied_y: bool, tmp_path: Any
) -> None:
    def build(ev: Any) -> Any:
        rec = _flat(lambda e: e.m * 3)(ev)
        if varied_y:
            rec = gak.with_field(rec, graphed.vary(ev.m, "s", points={"up": ev.m * 4, "same": ev.m}), "y")
        rec["k"] = scalar
        return rec

    manifest, _ = _write_and_check(build, tmp_path)
    values = [col for col in manifest["s_up"] if col.startswith("__vary_s_up__") and "__mask__" not in col]
    assert values == ["__vary_s_up__x", "__vary_s_up__y"][: 1 + varied_y]


def test_a_scalar_field_that_differs_by_universe_stores_a_delta(tmp_path: Any) -> None:
    def build(ev: Any) -> Any:
        rec = gak.zip({"m": ev.m}, depth_limit=1)
        k = {s: gak.with_field(rec, s, "k") for s in (1.5, 2.5)}
        return graphed.vary(k[1.5], "s", points={"up": k[2.5], "same": k[1.5]})

    manifest, _ = _write_and_check(build, tmp_path)
    assert manifest["s_up"]["__vary_s_up__k"]["representation"] == "xor"
