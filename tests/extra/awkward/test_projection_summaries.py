"""A memo entry's later nodes replay its first node's summary, or are rebuilt where it fails."""

from __future__ import annotations

import json
from typing import Any

import awkward as ak
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, project_buffers_many, projection
from graphed.awkward.projection import _buffers, _walked
from graphed.projection import BufferNeed
from graphed.systematics.by_label import cone

D = BufferNeed.DATA
EVENTS = {
    "x": [1.0, 2.0],
    "y": [3.0, 4.0],
    "Jet": [[{"pt": 30.0, "eta": 0.1}], []],
    "Pho": [[{"pt": 40.0, "eta": 1.1}], [{"pt": 25.0, "eta": -0.4}]],
}


def _events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    return session, from_awkward("ev", ak.Array(EVENTS), session=session)


def _counted(session: Session, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    evaluate = session.backend.eval_stage

    def counting(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        return evaluate(name, *args, **kwargs)

    monkeypatch.setattr(session.backend, "eval_stage", counting)
    return calls


def _universes(ev: Any, n: int) -> list[Any]:
    """``n`` universes scaling one collection each, then one selection over every universe."""
    out = []
    for i in range(n):
        jets = ev.Jet if i % 2 else ev.Pho
        jets = gak.with_field(jets, jets.pt * (1.0 + i / 100), "pt")
        good = jets[jets.pt > 20.0]
        out += [gak.sum(good.eta, axis=1), gak.num(good, axis=1)]
    return out


def test_a_repeated_entry_is_evaluated_once(monkeypatch: pytest.MonkeyPatch) -> None:
    session, ev = _events()
    first, second = ev.x + 1.0, ev.y + 1.0
    assert session._entries[first.node_id] == session._entries[second.node_id] >= 0
    union = {nid for out in (first, second) for nid in cone(session, out.node_id)}
    calls = _counted(session, monkeypatch)
    assert project_buffers_many([first, second]).read_buffers == {"ev": {"x": D, "y": D}}
    assert calls.count("add") == 1
    assert len(calls) == len(union - {ev.node_id}) - 1


def test_summaries_give_the_full_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    session, ev = _events()
    outputs = _universes(ev, 6)
    union = {nid for out in outputs for nid in cone(session, out.node_id)}
    full = _buffers(*_walked(outputs, "raise", summarize=False)).read_buffers
    calls = _counted(session, monkeypatch)
    assert project_buffers_many(outputs).read_buffers == full
    assert full == {"ev": {"Jet.pt": D, "Jet.eta": D, "Pho.pt": D, "Pho.eta": D}}
    assert len(calls) < len(union - {ev.node_id}) / 2


def test_a_source_buffer_where_the_summary_had_a_derived_one_is_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, ev = _events()
    first, second = ev.y * 2.0 + 1.0, ev.x + 1.0
    assert session._entries[first.node_id] == session._entries[second.node_id] >= 0
    union = {nid for out in (first, second) for nid in cone(session, out.node_id)}
    calls = _counted(session, monkeypatch)
    assert project_buffers_many([first, second]).read_buffers == {"ev": {"x": D, "y": D}}
    assert len(calls) == len(union - {ev.node_id})  # each node once: no whole-graph fallback


def test_another_alias_pattern_is_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    session, ev = _events()
    first, second = ev.x * ev.x, ev.x * ev.y
    assert session._entries[first.node_id] == session._entries[second.node_id] >= 0
    union = {nid for out in (first, second) for nid in cone(session, out.node_id)}
    calls = _counted(session, monkeypatch)
    assert project_buffers_many([first, second]).read_buffers == {"ev": {"x": D, "y": D}}
    assert len(calls) == len(union - {ev.node_id})


def test_a_summarized_output_reads_its_own_columns() -> None:
    session, ev = _events()
    first, second = gak.with_field(ev.Jet, ev.x, "w"), gak.with_field(ev.Pho, ev.y, "w")
    assert session._entries[first.node_id] == session._entries[second.node_id] >= 0
    assert project_buffers_many([first, second]).read_buffers == {
        "ev": {"x": D, "y": D, "Jet.pt": D, "Jet.eta": D, "Pho.pt": D, "Pho.eta": D}
    }


def test_a_scalar_output_is_summarized(monkeypatch: pytest.MonkeyPatch) -> None:
    session, ev = _events()
    outputs = [gak.sum(ev.x * 2.0, axis=None), gak.sum(ev.y * 2.0, axis=None)]
    calls = _counted(session, monkeypatch)
    assert project_buffers_many(outputs).read_buffers == {"ev": {"x": D, "y": D}}
    assert calls.count("mul") == calls.count("ak.sum") == 1


def test_an_external_reads_summarized_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    sf = {"name": "sf", "inputs": [{"name": "v", "type": "real"}]}
    cset = json.dumps({"schema_version": 2, "corrections": [sf]}).encode()
    session, ev = _events()
    first, second = ev.x[1:], ev.y[1:]  # a view touches nothing itself
    outputs = [first, gak.apply_correction(cset, "sf", v=second)]
    calls = _counted(session, monkeypatch)
    projection = project_buffers_many(outputs, on_fail="pass")
    assert projection.read_buffers == {"ev": {"x": D, "y": D}}
    assert calls.count("slice") == 1


def test_a_rebuilt_node_reading_one_input_twice_keeps_both_pass_throughs() -> None:
    session, ev = _events()
    first, second = ev.x[1:], ev.y[1:]
    doubled = first * 2.0
    learned = gak.zip({"a": doubled, "b": doubled})
    rebuilt = gak.zip({"a": second, "b": second})
    assert session._entries[learned.node_id] == session._entries[rebuilt.node_id] >= 0
    outputs = [first, learned.a, rebuilt.a]
    full = _buffers(*_walked(outputs, "raise", summarize=False)).read_buffers
    assert full == {"ev": {"x": D, "y": D}}
    assert project_buffers_many(outputs).read_buffers == full


def test_a_rebuilt_node_touches_the_sources_its_summarized_inputs_hold() -> None:
    _, ev = _events()
    first = (ev.y * 2.0)[1:] + 1.0
    second = ev.x[1:] + 1.0  # both nodes rebuilt; the add alone reads x
    assert project_buffers_many([first, second]).read_buffers == {"ev": {"x": D, "y": D}}


def _reaching(monkeypatch: pytest.MonkeyPatch, session: Session, outside: Any) -> list[str]:
    """Make the first ``add`` touch, or return, ``outside(sources)`` beyond its own inputs."""
    tracers: dict[int, Any] = {}
    sources = projection._sources
    monkeypatch.setattr(
        projection, "_sources", lambda s: (r := sources(s), tracers.update(r[1]))[0]
    )
    calls = _counted(session, monkeypatch)
    evaluate = session.backend.eval_stage

    def reaching(name: str, ins: list[Any], params: Any) -> Any:
        out = evaluate(name, ins, params)
        return outside(next(iter(tracers.values())), out) if calls.count("add") == 1 else out

    monkeypatch.setattr(session.backend, "eval_stage", reaching)
    return calls


@pytest.mark.parametrize(
    "outside",
    [
        lambda ev, out: (ak.typetracer.touch_data(ev.y), out)[1],  # a touch beyond its inputs
        lambda ev, out: ev.y,  # an output buffer none of its inputs holds
    ],
)
def test_a_node_reaching_beyond_its_inputs_is_not_summarized(
    monkeypatch: pytest.MonkeyPatch, outside: Any
) -> None:
    session, ev = _events()
    first, second = ev.x + 1.0, ev.y + 1.0
    assert session._entries[first.node_id] == session._entries[second.node_id] >= 0
    calls = _reaching(monkeypatch, session, outside)
    project_buffers_many([first, second])
    assert calls.count("add") == 2
