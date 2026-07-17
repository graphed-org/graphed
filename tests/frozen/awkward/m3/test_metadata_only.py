"""'Metadata only' is truly enforced: recording never reads event data (plan M3)."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
from analyses import ADL, record
from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward, from_parquet, gak


def test_recording_keeps_every_form_on_the_typetracer_backend() -> None:
    s, _ = record(ADL["q6"])
    for node_id, form in s._forms.items():
        assert form.is_typetracer, f"node {node_id} form left the typetracer (data was read)"


def test_op_form_does_not_touch_data() -> None:
    # A typetracer raises if concrete values are read; successful recording proves metadata-only.
    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=100))
    jets = events.Jet[events.Jet.pt > 30]
    ht = gak.sum(jets.pt, axis=1)
    assert s.form(ht).is_typetracer


def test_from_parquet_reads_only_metadata(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    ak.to_parquet(make_events(n_events=500), str(path))
    s = Session(AwkwardBackend())
    events = from_parquet(s, "events", str(path))
    # the form is available from metadata alone (no row groups read)
    muon_pt = events.Muon.pt
    assert s.form(muon_pt).is_typetracer
    assert "var * float64" in s.form(muon_pt).describe()
    # and the data is still materializable lazily (read only now)
    njet = gak.num(events.Jet, axis=1)
    result = s.materialize(njet)
    assert len(ak.Array(result)) == 500
