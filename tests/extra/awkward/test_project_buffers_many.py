"""`project_buffers_many` answers the union of several outputs' buffer needs from ONE reporting
typetracer per source, where a per-output `project_buffers` loop builds one per source per output."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import BufferNeed, BufferProjection, Session
from graphed.awkward import AwkwardBackend, from_awkward, gak, project_buffers, project_buffers_many

EVENTS = ak.Array(
    {
        "Jet": [[{"pt": 50.0, "eta": 0.1}, {"pt": 30.0, "eta": 2.2}], [], [{"pt": 70.0, "eta": -0.5}]],
        "MET": [12.0, 35.0, 8.0],
    }
)
OTHER = ak.Array({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


def _outputs(opaque: bool = False) -> tuple[Session, list[Any]]:
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    other = from_awkward(session, "other", OTHER)
    # a count-only need on Jet next to a DATA need under it: the union's covering rule is exercised
    outputs = [events.MET * 2, gak.num(events.Jet, axis=1), events.Jet.pt + 1, other.x + events.MET]
    if opaque:
        outputs.append(other.y.map(lambda y: y))
    return session, outputs


def _union(projections: Sequence[BufferProjection]) -> dict[str, dict[str, BufferNeed]]:
    """Per-output needs merged by the module's rule: DATA wins, and an OFFSETS path is dropped when a
    DATA path at or under it already brings its structure."""
    merged: dict[str, dict[str, BufferNeed]] = {}
    for projection in projections:
        for source, needs in projection.read_buffers.items():
            into = merged.setdefault(source, {})
            for path, need in needs.items():
                if into.get(path) is not BufferNeed.DATA:
                    into[path] = need
    for needs in merged.values():
        data = [p for p, n in needs.items() if n is BufferNeed.DATA]
        for path in [p for p, n in needs.items() if n is BufferNeed.OFFSETS]:
            if any(d == path or d.startswith(path + ".") for d in data):
                del needs[path]
    return merged


@pytest.fixture
def replays(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[object]]:
    calls: list[object] = []
    real = ak.typetracer.typetracer_with_report

    def counting(form: object, **kwargs: Any) -> Any:
        calls.append(form)
        return real(form, **kwargs)

    monkeypatch.setattr(ak.typetracer, "typetracer_with_report", counting)
    yield calls


@pytest.mark.parametrize("on_fail", ["raise", "pass", "warn"])
def test_many_equals_the_union_of_per_output_projections(on_fail: str) -> None:
    opaque = on_fail != "raise"
    _, outputs = _outputs(opaque=opaque)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        many = project_buffers_many(outputs, on_fail=on_fail)
        each = [project_buffers(output, on_fail=on_fail) for output in outputs]
    assert isinstance(many, BufferProjection)
    assert many.read_buffers == _union(each)
    assert many.buffers_for("events") != {} and many.buffers_for("other") != {}


def test_the_count_only_need_survives_when_nothing_under_it_is_read() -> None:
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    many = project_buffers_many([gak.num(events.Jet, axis=1), events.MET])
    assert many.buffers_for("events") == {"Jet": BufferNeed.OFFSETS, "MET": BufferNeed.DATA}


def test_an_opaque_output_under_warn_makes_every_source_conservative() -> None:
    _, outputs = _outputs(opaque=True)
    with pytest.warns(UserWarning):
        many = project_buffers_many(outputs, on_fail="warn")
    assert many.buffers_for("events") == dict.fromkeys(("Jet.pt", "Jet.eta", "MET"), BufferNeed.DATA)
    assert many.buffers_for("other") == dict.fromkeys(("x", "y"), BufferNeed.DATA)


def test_one_reporting_typetracer_per_source(replays: list[object]) -> None:
    session, outputs = _outputs()
    sources = len(session.source_ids())
    project_buffers_many(outputs)
    assert len(replays) == sources
    replays.clear()
    for output in outputs:
        project_buffers(output)
    assert len(replays) == sources * len(outputs)


def test_project_buffers_is_the_single_output_case() -> None:
    _, outputs = _outputs()
    for output in outputs:
        assert project_buffers(output).read_buffers == project_buffers_many([output]).read_buffers


def test_an_output_from_another_session_is_refused() -> None:
    _, outputs = _outputs()
    _, foreign = _outputs()
    with pytest.raises(TypeError, match="different Session"):
        project_buffers_many([*outputs, foreign[0]])


def test_no_outputs_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one output"):
        project_buffers_many([])


def test_a_varied_member_expands_per_label() -> None:
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    jets = events.Jet
    varied = graphed.vary(jets.pt, "jes", points={"up": jets.pt * 1.05, "down": jets.pt * 0.95})
    met = events.MET * 2
    answer = project_buffers_many([varied, met])
    assert tuple(answer) == graphed.labels(varied)
    for label, projection in answer.items():
        member = graphed.universe(varied, label)
        assert projection.read_buffers == project_buffers_many([member, met]).read_buffers
