"""M19: introspection + peeking conveniences (parity plan P3.8 — the LAST MVP tier).

`fields`/`type_of`/`backend_of` answer from the recorded FORM and the session — pure metadata,
recording NOTHING (witnessed by node counts). `head`/`sample` are EAGER peeking sugar over the
common slice op + the reference materialize."""

from __future__ import annotations

import os

import awkward as ak
import pytest
from graphed import Session
from m16_helpers import EVENTS, session_events

from graphed_awkward import AwkwardBackend, from_parquet, gak


def test_fields_and_type_answer_from_the_form_without_recording() -> None:
    s, g, _ = session_events()
    n = s.node_count()
    assert gak.fields(g) == ["Jet", "met"]
    assert gak.fields(g.Jet) == ["pt", "eta"]
    assert "var *" in gak.type_of(g.Jet.pt)
    assert gak.backend_of(g) == "AwkwardBackend"
    assert s.node_count() == n + 2  # only the two field accesses above; the introspection: zero


def test_head_returns_the_first_rows_eagerly() -> None:
    _s, g, raw = session_events()
    got = ak.Array(gak.head(g.met, 2))
    assert ak.array_equal(got, raw.met[:2])
    assert len(ak.Array(gak.head(g.Jet, 3))) == 3


def test_sample_takes_every_kth_row() -> None:
    _s, g, raw = session_events()
    got = ak.Array(gak.sample(g.met, factor=2))
    assert ak.array_equal(got, raw.met[::2])


def test_head_works_on_parquet_sources(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pyarrow")
    ak.to_parquet(EVENTS, os.path.join(tmp_path, "e.parquet"))
    s = Session(AwkwardBackend())
    g = from_parquet(s, "events", os.path.join(tmp_path, "e.parquet"))
    got = ak.Array(gak.head(g.met, 2))
    assert ak.array_equal(got, EVENTS.met[:2])
