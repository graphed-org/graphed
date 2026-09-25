"""m71a A-C2/A-C7 (awkward): the canonical declared type is node identity — spellings of one type
intern to one node, different types never collapse, and the IR carries the canonical string."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
from awkward.types import ListType, NumpyType, OptionType, RecordType, RegularType
from m71a_awkward_fixtures import (
    NUMERIC,
    PHOTON,
    byte_strings,
    describe,
    eager,
    f,
    jet_photons,
    masked_jets,
    pair,
    params_of,
    photon,
    recorded,
    same,
    strings,
    triples,
)

from graphed.awkward import gak

F32 = NumpyType("float32")
PAIR = RecordType([F32, F32], ["pt", "eta"])
MASKED = ListType(OptionType(RecordType([F32, NumpyType("int64")], ["pt", "idx"])))
TRACER = ak.Array(eager().Jet.pt.layout.to_typetracer(forget_length=True))


def _float16(a: Any) -> Any:
    return ak.values_astype(a, "float16")


#: (canonical, spellings, column, callable, name, materialize-and-compare)
FAMILIES: list[tuple[str, list[Any], str, Callable[[Any], Any], str, bool]] = [
    ("float32", ["float32", np.float32, np.dtype("f4"), "f4", F32], "x", lambda a: a, "fam", False),
    ("bool", [bool, np.bool_, "bool", NumpyType("bool")], "x", f, "fam", False),
    ("int64", [int, np.int64, "i8"], "x", lambda a: ak.values_astype(a, "int64"), "fam", False),
    ("float64", [float, "float64"], "x", lambda a: ak.values_astype(a, "float64"), "fam", False),
    ("complex128", [complex, "complex128"], "x", lambda a: ak.values_astype(a, "complex128"), "fam", False),
    ("string", ["string", "U5", np.dtype("U5"), str], "x", strings, "fam", True),
    ("bytes", ["bytes", "S5", bytes], "x", byte_strings, "fam", True),
    ("float16", [np.float16, "f2", "float16"], "x", _float16, "h", True),
    ("3 * float32", ["3 * float32", ("f4", (3,)), RegularType(F32, 3)], "x", triples, "fam", True),
    (
        "{pt: float32, eta: float32}",
        [
            "{pt: float32, eta: float32}",
            PAIR,
            [("pt", "f4"), ("eta", "f4")],
            np.dtype([("pt", "f4"), ("eta", "f4")]),
            ak.zip({"pt": np.zeros(1, np.float32), "eta": np.zeros(1, np.float32)})[0].type,
        ],
        "x",
        pair,
        "fam",
        True,
    ),
    (
        PHOTON,
        [PHOTON, RecordType([F32, F32], ["pt", "eta"], parameters={"__record__": "Photon"})],
        "x",
        photon,
        "fam",
        True,
    ),
    (
        "var * float32",
        ["var * float32", "var*float32", ListType(F32)],
        "pt",
        lambda p: ak.values_astype(p, "float32"),
        "famj",
        False,
    ),
    (
        "var * ?{pt: float32, idx: int64}",
        ["var * ?{pt: float32, idx: int64}", MASKED],
        "pt",
        masked_jets,
        "famj",
        True,
    ),
    (
        "var * float64",
        [TRACER.type, TRACER.type.content, str(TRACER.type.content)],
        "pt",
        lambda p: p,
        "famj",
        False,
    ),
]


def _column(ev: Any, column: str) -> Any:
    return ev.x if column == "x" else ev.Jet.pt


def test_declared_and_undeclared_are_two_nodes_with_two_forms() -> None:
    s, ev = recorded()
    declared = ev.Jet.pt.map(f, name="j", output_type="var * bool")
    undeclared = ev.Jet.pt.map(f, name="j")

    assert declared.node_id != undeclared.node_id
    assert describe(s, declared) == "## * var * bool"
    assert describe(s, undeclared) == "## * var * float64"
    assert "output_type" not in params_of(s, undeclared)


def test_two_declared_types_are_two_nodes() -> None:
    s, ev = recorded()
    as_bool = ev.Jet.pt.map(f, name="j", output_type="var * bool")
    as_int8 = ev.Jet.pt.map(f, name="j", output_type="var * int8")

    assert as_bool.node_id != as_int8.node_id
    assert describe(s, as_int8) == "## * var * int8"


def test_spellings_of_one_type_are_one_node_carrying_the_canonical_string() -> None:
    s, ev = recorded()
    obj = ak.types.from_datashape("var * bool", highlevel=False)
    nodes = [
        ev.Jet.pt.map(f, name="j", output_type=spec)
        for spec in ("var*bool", "var * bool", obj, ak.forms.from_type(obj))
    ]

    assert len({n.node_id for n in nodes}) == 1
    assert params_of(s, nodes[0])["output_type"] == "var * bool"
    assert b"var * bool" in s.serialized_ir(nodes[0], optimize=False)
    assert b"var*bool" not in s.serialized_ir(nodes[0], optimize=False)


def test_each_spelling_family_is_one_node_with_the_canonical_string() -> None:
    s, ev = recorded()
    e = eager()
    for canonical, spellings, column, fn, name, check in FAMILIES:
        arrays = [_column(ev, column).map(fn, name=name, output_type=spec) for spec in spellings]
        assert len({a.node_id for a in arrays}) == 1, canonical
        assert params_of(s, arrays[0])["output_type"] == canonical
        assert canonical.encode() in s.serialized_ir(arrays[0], optimize=False)
        assert describe(s, arrays[0]) == f"## * {canonical}"
        if check:
            assert same(s.materialize(arrays[0]), fn(_column(e, column))), canonical


def test_spelling_families_are_pairwise_distinct_nodes() -> None:
    _s, ev = recorded()
    ids = {
        _column(ev, column).map(fn, name=name, output_type=spellings[0]).node_id
        for _canonical, spellings, column, fn, name, _check in FAMILIES
    }
    assert len(ids) == len(FAMILIES)


def test_a_python_int_is_int64_on_every_platform() -> None:
    s, ev = recorded()
    assert params_of(s, ev.x.map(f, name="fam", output_type=int))["output_type"] == "int64"


def test_every_numerical_dtype_is_declarable() -> None:
    s, ev = recorded()
    e = eager()
    ids = set()
    for name in NUMERIC:
        fn: Callable[[Any], Any] = lambda a, name=name: ak.values_astype(a, name)  # noqa: E731
        scalar_type = np.bool_ if name == "bool" else getattr(np, name)
        arrays = [ev.x.map(fn, name="num", output_type=spec) for spec in (name, np.dtype(name), scalar_type)]
        assert len({a.node_id for a in arrays}) == 1, name
        assert params_of(s, arrays[0])["output_type"] == name
        assert describe(s, arrays[0]) == f"## * {name}"
        assert same(s.materialize(arrays[0]), fn(e.x)), name
        ids.add(arrays[0].node_id)
    assert len(ids) == len(NUMERIC)


def _program() -> bytes:
    s, ev = recorded()
    outs = [
        ev.run.map(f, name="m", output_type="bool"),
        ev.Jet.pt.map(f, name="j", output_type="var * bool"),
        ev.Jet.pt.map(f, name="j"),
        ev.Jet.pt.map(f, name="j", output_type="var * int8"),
        ev.x.map(pair, name="r", output_type="{pt: float32, eta: float32}"),
        ev.Jet.pt.map(masked_jets, name="o", output_type="var * ?{pt: float32, idx: int64}"),
        ev.Jet.pt.map(jet_photons, name="g", output_type=f"var * {PHOTON}").pt2,
        gak.sum(ev.x).map(f, name="s", output_type="bool"),
    ]
    for _canonical, spellings, column, fn, name, _check in FAMILIES:
        outs.extend(_column(ev, column).map(fn, name=name, output_type=spec) for spec in spellings)
    outs.extend(ev.x.map(f, name="num", output_type=name) for name in NUMERIC)
    return bytes(s.serialized_ir(*outs, optimize=False))


def test_declared_recordings_serialize_byte_identically_across_sessions() -> None:
    first, second = _program(), _program()
    assert first == second
    assert b"var * ?{pt: float32, idx: int64}" in first and b"float16" in first
