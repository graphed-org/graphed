"""M18: awkward behaviors — the HEP four-vector machinery, lazily (parity plan P2).

`gak.with_name` + a behavior dict registered on the backend make behavior PROPERTIES (`.pt`,
`.mass`) work through plain attribute access: the typetracer evaluates them at record time
(metadata only), evaluation computes them, and — the truthfulness pin — the buffer projection
reports exactly the leaf columns the property actually reads. Functions + attribute access only:
no proxy changes (the factorization rule).
"""

from __future__ import annotations

import awkward as ak
import pytest
from graphed import GraphedTypeError, Session

vector = pytest.importorskip("vector")
vector.register_awkward()

from graphed_awkward import AwkwardBackend, from_awkward, gak, project  # noqa: E402

MUONS = ak.Array(
    {
        "Muon": [
            [{"px": 1.0, "py": 2.0, "pz": 0.5, "E": 3.0}, {"px": -1.0, "py": 0.5, "pz": 2.0, "E": 4.0}],
            [],
            [{"px": 3.0, "py": -2.0, "pz": 1.0, "E": 5.0}],
        ]
    }
)


def _ref() -> ak.Array:
    return ak.Array(ak.with_name(MUONS.Muon, "Momentum4D").layout, behavior=vector.backends.awkward.behavior)


def _s() -> tuple[Session, object]:
    s = Session(AwkwardBackend(behavior=vector.backends.awkward.behavior))
    return s, from_awkward(s, "events", MUONS)


def test_behavior_properties_record_with_typetracer_forms() -> None:
    s, g = _s()
    v = gak.with_name(g.Muon, "Momentum4D")  # type: ignore[attr-defined]
    pt = v.pt  # a behavior PROPERTY, not a record field
    form = s.form(pt)
    assert form.is_typetracer  # inferred from metadata alone
    assert "var * float64" in form.describe()


def test_behavior_properties_evaluate_exactly() -> None:
    s, g = _s()
    v = gak.with_name(g.Muon, "Momentum4D")  # type: ignore[attr-defined]
    ref = _ref()
    assert ak.array_equal(ak.Array(s.materialize(v.pt)), ref.pt)
    assert ak.array_equal(ak.Array(s.materialize(v.mass)), ref.mass, equal_nan=True)
    assert ak.array_equal(ak.Array(s.materialize(v.px)), ref.px)  # plain fields still fields


def test_projection_reports_what_the_behavior_property_actually_reads() -> None:
    _session, g = _s()
    v = gak.with_name(g.Muon, "Momentum4D")  # type: ignore[attr-defined]
    out = gak.sum(v.pt, axis=1)
    # pt = hypot(px, py): exactly those two leaves, nothing else — behaviors stay projectable
    assert project(out).columns_for("events") == frozenset({"Muon.px", "Muon.py"})


def test_unknown_attributes_fail_at_record_time() -> None:
    s = Session(AwkwardBackend())  # no behavior registered
    g = from_awkward(s, "events", MUONS)
    with pytest.raises(GraphedTypeError):
        _ = g.Muon.pt  # not a field, and no behavior supplies it


def test_with_parameter_and_without_parameters() -> None:
    s, g = _s()
    tagged = gak.with_parameter(g.Muon, "origin", "simulation")  # type: ignore[attr-defined]
    out = ak.Array(s.materialize(tagged))
    assert out.layout.parameter("origin") == "simulation"
    cleared = gak.without_parameters(tagged)  # type: ignore[attr-defined]
    out2 = ak.Array(s.materialize(cleared))
    assert out2.layout.parameter("origin") is None
    assert s.form(cleared).is_typetracer


def test_named_records_survive_structure_ops() -> None:
    s, g = _s()
    v = gak.with_name(g.Muon, "Momentum4D")  # type: ignore[attr-defined]
    leading = gak.firsts(v, axis=1)
    got = ak.Array(s.materialize(leading.pt))
    ref = ak.firsts(_ref(), axis=1).pt
    assert ak.array_equal(got, ref, equal_nan=True)
