"""m71a A-C1/A-C3 (awkward): the declared type is the recorded form, so build-time ops type-check
against it — a bool mask indexes as a mask, and a named record resolves its behavior."""

from __future__ import annotations

import awkward as ak
from m71a_awkward_fixtures import (
    PHOTON,
    describe,
    eager,
    f,
    jet_photons,
    masked_jets,
    pair,
    recorded,
    same,
)

import graphed
from graphed.awkward import gak, project


def test_a_declared_bool_mask_indexes_as_a_mask() -> None:
    s, ev = recorded()
    e = eager()
    m = ev.run.map(f, name="m", output_type="bool")

    assert describe(s, m) == "## * bool"
    assert describe(s, ~m) == "## * bool"
    assert describe(s, ev.x[m]) == "## * float32"
    assert same(s.materialize(m), f(e.run))
    assert same(s.materialize(ev.x[m]), e.x[f(e.run)])


def test_a_declared_jagged_mask_indexes_as_a_jagged_mask() -> None:
    s, ev = recorded()
    e = eager()
    jm = ev.Jet.pt.map(f, name="j", output_type="var * bool")

    assert describe(s, jm) == "## * var * bool"
    assert describe(s, ev.Jet.pt[jm]) == "## * var * float64"
    assert same(s.materialize(ev.Jet.pt[jm]), e.Jet.pt[f(e.Jet.pt)])


def test_the_declared_type_is_not_a_leaf_cast_of_the_input() -> None:
    s, ev = recorded()
    e = eager()
    n = ev.Jet.pt.map(lambda p: ak.num(p) > 1, name="n", output_type="bool")

    assert describe(s, n) == "## * bool"
    assert same(s.materialize(n), ak.num(e.Jet.pt) > 1)


def test_a_declared_record_exposes_typed_fields() -> None:
    s, ev = recorded()
    e = eager()
    r = ev.x.map(pair, name="r", output_type="{pt: float32, eta: float32}")

    assert describe(s, r) == "## * {pt: float32, eta: float32}"
    assert describe(s, r.eta) == "## * float32"
    assert same(s.materialize(r.eta), pair(e.x).eta)


def test_a_declared_option_of_records_exposes_optional_fields() -> None:
    s, ev = recorded()
    e = eager()
    o = ev.Jet.pt.map(masked_jets, name="o", output_type="var * ?{pt: float32, idx: int64}")

    assert describe(s, o) == "## * var * ?{pt: float32, idx: int64}"
    assert describe(s, o.idx) == "## * var * ?int64"
    assert same(s.materialize(o), masked_jets(e.Jet.pt))
    assert same(s.materialize(o.idx), masked_jets(e.Jet.pt).idx)


def test_a_scalar_input_records_the_declared_scalar() -> None:
    s, ev = recorded()
    e = eager()
    out = gak.sum(ev.x).map(f, name="s", output_type="bool")

    assert describe(s, out) == "bool"
    assert s.materialize(out) == f(ak.sum(e.x))


def test_every_universe_of_a_varied_apply_records_the_declared_type() -> None:
    s, ev = recorded()
    out = graphed.apply(
        f, graphed.vary(ev.x, "jes", up=ev.x * 1.5, down=ev.x * 0.5), name="v", output_type="bool"
    )

    labels = graphed.labels(out)
    assert len(labels) == 3
    assert {describe(s, graphed.universe(out, lbl)) for lbl in labels} == {"## * bool"}


def test_a_declared_named_record_resolves_its_behavior() -> None:
    s, ev = recorded()
    e = eager()
    g = ev.Jet.pt.map(jet_photons, name="g", output_type=f"var * {PHOTON}")
    expected = ak.Array(jet_photons(e.Jet.pt), behavior=e.behavior)

    assert describe(s, g) == f"## * var * {PHOTON}"
    assert describe(s, g.pt2) == "## * var * float32"
    assert describe(s, g.scaled(3)) == "## * var * float32"
    assert same(s.materialize(g.pt2), expected.pt2)
    assert same(s.materialize(g.scaled(3)), expected.scaled(3))
    assert set(project(g.pt2, on_fail="pass").columns_for("events")) == {"Jet.pt"}
