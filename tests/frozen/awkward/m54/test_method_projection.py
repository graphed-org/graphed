"""m54 §2.6 — a `method` op is partition-local, fusible, projectable, and backend-sourced.

The two halves are the two ways a new op kind goes wrong. Projection: an op that could not be
replayed through the reporting typetracer would fall back to reading the whole source, so the
assertion is EQUALITY against an independently measured eager oracle (over-read fails it as loudly
as under-read), plus the leaf the method provably never reads. Fusion: an op mistakenly treated as a
boundary would split the run in two, which the reduced store's node kinds show directly.
"""

from __future__ import annotations

import awkward as ak
from m54_behavior_fixtures import (
    BEHAVIOR,
    EVENTS,
    attrs_sources,
    eager,
    jets_source,
    recorded,
    source_id,
    top_level,
    touched_columns,
)

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, project


def _lead_delta_r(events: ak.Array) -> ak.Array:
    """The eager twin of the recorded program below, for the oracle."""
    jets = ak.with_name(events.Jet, "Jet")
    probe = ak.with_name(events.Probe, "Jet")
    return ak.firsts(jets, axis=1).deltaR(ak.firsts(probe, axis=1))


def test_projection_reports_exactly_the_leaves_the_method_reads() -> None:
    _session, jets, probe = recorded()
    out = gak.sum(gak.firsts(jets, axis=1).deltaR(gak.firsts(probe, axis=1)))

    columns = set(project(out).columns_for("events"))
    assert columns == touched_columns(_lead_delta_r)
    assert "Jet.mass" not in columns  # deltaR is angular; mass is never read
    assert {"Jet.pt", "Jet.eta", "Jet.phi"} <= columns


def test_a_method_does_not_end_a_stage() -> None:
    """The property analogue of the same program reduces to source + one stage + reduction; a
    boundary op would make it two stages, or emit a node kind of its own."""
    session, jets, probe = recorded()
    method_out = gak.sum(gak.firsts(jets, axis=1).deltaR(gak.firsts(probe, axis=1)))
    property_out = gak.sum(gak.firsts(jets, axis=1).rho - gak.firsts(probe, axis=1).rho)

    def kinds(node_id: int) -> list[str]:
        reduced, _report = session._store.reduce(outputs=[node_id])
        return [node["kind"] for node in reduced.nodes()]

    assert kinds(property_out.node_id) == ["source", "stage", "reduction"]
    assert kinds(method_out.node_id) == ["source", "stage", "reduction"]


def test_a_backend_only_behavior_resolves_like_a_globally_registered_one() -> None:
    """`JetArray` and the `Jet` ufunc overloads live in the backend's dict alone. If the backend
    handed bare arrays to the typetracer or to evaluation, only vector's globally registered names
    would work — so these three legs together separate the two registration routes."""
    assert ("*", "Jet") not in ak.behavior
    assert "Jet" not in ak.behavior

    session, jets, _probe = recorded()
    assert ak.array_equal(
        ak.Array(session.materialize(jets.scaled(2.0))), eager("Jet").scaled(2.0)
    )
    assert ak.array_equal(ak.Array(session.materialize(jets.rho)), eager("Jet").rho)
    assert ak.array_equal(
        ak.Array(session.materialize((jets + jets).mass)),
        (eager("Jet") + eager("Jet")).mass,
        equal_nan=True,
    )


def test_the_read_list_never_under_reads_for_a_field_rooted_method() -> None:
    """`read_columns` drives I/O at TOP-LEVEL field granularity, so the gate is prefix containment:
    every leaf the method touches must have its field in the read list. The leaf-level narrowing is
    projection's job, so the `Jet.*` half of that is asserted against the same oracle."""
    session, jets, probe = recorded()
    out = gak.sum(gak.firsts(jets, axis=1).deltaR(gak.firsts(probe, axis=1)))

    touched = touched_columns(_lead_delta_r)
    columns = graphed.read_columns([out], source_id(session))
    assert columns is None or set(columns) >= top_level(touched)

    narrowed = project(out).columns_for("events")
    assert {c for c in narrowed if c.startswith("Jet.")} == {
        leaf for leaf in touched if leaf.startswith("Jet.")
    }


def test_the_read_list_never_under_reads_for_a_method_on_the_source_record() -> None:
    """A method applied directly to the source record may read the whole source; what it may NOT do
    is answer a strict subset of what the method touches. `None` stays legal, and so does a later
    narrowing — hence the superset, not an equality or an `is None`."""
    _session, source, source_nid = jets_source()
    out = gak.sum(source.scaled(2.0))

    touched = top_level(touched_columns(lambda jets: jets.scaled(2.0), source=EVENTS.Jet))
    assert touched  # the oracle is live, or the containment below is vacuous

    columns = graphed.read_columns([out], source_nid)
    assert columns is None or set(columns) >= touched


def test_the_behavior_re_wrap_carries_the_arrays_attrs() -> None:
    """A re-wrap built from the layout alone loses `attrs`, so both members answer their default —
    half the value here. Pinned for a property and a method, on a source whose records are already
    named and on one named through `gak.with_name`."""
    named, events = attrs_sources()

    control = Session(AwkwardBackend())
    plain = from_awkward(control, "jets", named)
    assert ak.array_equal(ak.Array(control.materialize(plain.pt)), named.pt)

    on_source_session = Session(AwkwardBackend(behavior=BEHAVIOR))
    on_source = from_awkward(on_source_session, "jets", named)
    renamed_session = Session(AwkwardBackend(behavior=BEHAVIOR))
    renamed = gak.with_name(from_awkward(renamed_session, "events", events).Jet, "Jet")

    for session, jets in ((on_source_session, on_source), (renamed_session, renamed)):
        assert ak.array_equal(
            ak.Array(session.materialize(jets.calibrated)), named.calibrated
        )
        assert ak.array_equal(
            ak.Array(session.materialize(jets.calibrate(3.0))), named.calibrate(3.0)
        )
