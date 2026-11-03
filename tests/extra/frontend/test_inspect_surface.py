"""#146: one-argument `read_columns`, `explain_variations`, and did-you-mean on a typo."""

from __future__ import annotations

from typing import Any

import awkward as ak
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.awkward.backend import AwkwardForm
from graphed.errors import GraphedError, GraphedTypeError, MissingFieldError


class MuonArray(ak.Array):  # type: ignore[misc]
    @property
    def energy(self) -> Any:
        return self.pt


BEHAVIOR: dict[Any, Any] = {("*", "Muon"): MuonArray}
EVENTS = ak.Array(
    {
        "MET_pt": [1.0, 2.0],
        "Muon_pt": [[1.0], []],
        "Muon": ak.with_name(ak.Array([[{"pt": 1.0, "eta": 0.5}], []]), "Muon"),
    }
)


def _events() -> Any:
    return from_awkward("Events", EVENTS, session=Session(AwkwardBackend(behavior=BEHAVIOR)))


def test_one_array_answers_by_source_name() -> None:
    ev = _events()
    assert graphed.read_columns(ev.MET_pt + 1.0) == {"Events": ("MET_pt",)}


def test_a_list_unions_its_columns() -> None:
    ev = _events()
    assert graphed.read_columns([ev.MET_pt, ev.Muon_pt]) == {"Events": ("MET_pt", "Muon_pt")}


def test_two_sources_each_answer_for_themselves() -> None:
    ev = _events()
    session = ev.session
    other = from_awkward("Other", ak.Array({"x": [1.0, 2.0], "y": [3.0, 4.0]}), session=session)
    assert graphed.read_columns([ev.MET_pt, other.x]) == {
        "Events": ("MET_pt",),
        "Other": ("x",),
    }


def test_a_whole_record_reads_every_column() -> None:
    ev = _events()
    assert graphed.read_columns(ev) == {"Events": None}


def test_a_varied_array_answers_for_every_universe() -> None:
    ev = _events()
    varied = graphed.vary(ev.MET_pt, "jes", up=ev.MET_pt + 1.0)
    assert graphed.read_columns(varied) == {"Events": ("MET_pt",)}


def test_the_source_id_form_still_answers_for_one_source() -> None:
    ev = _events()
    session = ev.session
    assert graphed.read_columns([ev.MET_pt], session.source_ids()[0]) == ("MET_pt",)
    assert graphed.read_columns(ev.MET_pt, session.source_ids()[0]) == ("MET_pt",)


def test_explain_is_a_deprecated_alias_of_explain_variations() -> None:
    session = Session(AwkwardBackend())
    ctx = graphed.awkward.gnano.events(from_awkward("Events", EVENTS, session=session))
    assert str(graphed.explain_variations(ctx)).startswith("graphed.explain_variations: 0")
    with pytest.warns(DeprecationWarning, match="use graphed.explain_variations"):
        assert graphed.explain(ctx) == graphed.explain_variations(ctx)


def test_explain_variations_refuses_what_is_not_a_context() -> None:
    ev = _events()
    with pytest.raises(GraphedError, match="explain_variations reads an event context"):
        graphed.explain_variations(ev)


@pytest.mark.parametrize(
    ("read", "message"),
    [
        (lambda ev: ev.MET_ptt, "no field named 'MET_ptt' (did you mean 'MET_pt'?)"),
        (lambda ev: ev.MET_pt.fieldz, "no field named 'fieldz' (did you mean 'fields'?)"),
        (lambda ev: ev.Muon.ptt, "no field named 'ptt' in Muon (did you mean 'pt'?)"),
        (lambda ev: ev.Muon.energi, "no field named 'energi' in Muon (did you mean 'energy'?)"),
    ],
)
def test_a_typo_suggests_a_close_name(read: Any, message: str) -> None:
    ev = _events()
    with pytest.raises(MissingFieldError) as caught:
        read(ev)
    assert message in str(caught.value)


def test_no_close_name_means_no_suggestion() -> None:
    ev = _events()
    with pytest.raises(MissingFieldError) as caught:
        ev.zzzzzz  # noqa: B018
    assert "no field named 'zzzzzz'" in str(caught.value)
    assert "did you mean" not in str(caught.value)


def test_form_points_at_typestr() -> None:
    ev = _events()
    with pytest.raises(MissingFieldError, match=r"`\.typestr` shows an array's type"):
        ev.MET_pt.form  # noqa: B018


def test_a_record_field_named_form_still_reads() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("Events", ak.Array({"form": [1.0]}), session=session)
    assert ak.to_list(session.materialize(ev.form)) == [1.0]


def test_attribute_kind_says_what_is_missing_to_a_direct_caller() -> None:
    ev = _events()
    session = ev.session
    form: AwkwardForm = session.form(ev.Muon)
    with pytest.raises(AttributeError, match=r"no field named 'ptt' in Muon \(did you mean 'pt'"):
        session.backend.attribute_kind(form, "ptt")


def test_a_name_shared_across_sessions_answers_for_both_sources() -> None:
    ev = _events()
    other = _events()
    assert graphed.read_columns([ev.MET_pt, other.Muon_pt]) == {"Events": ("MET_pt", "Muon_pt")}
    assert graphed.read_columns([ev, other.Muon_pt]) == {"Events": None}


def test_a_generator_is_as_good_as_a_list() -> None:
    ev = _events()
    session = ev.session
    assert graphed.read_columns(a for a in [ev.MET_pt]) == {"Events": ("MET_pt",)}
    assert graphed.read_columns((a for a in [ev.MET_pt]), session.source_ids()[0]) == ("MET_pt",)


def test_an_event_context_is_one_output() -> None:
    session = Session(AwkwardBackend())
    ctx = graphed.awkward.gnano.events(from_awkward("Events", EVENTS, session=session))
    assert graphed.read_columns(ctx) == {"Events": None}


def test_a_string_read_says_what_is_missing_too() -> None:
    with pytest.raises(
        GraphedTypeError, match=r"no field named 'MET_ptt' .*did you mean 'MET_pt'"
    ):
        _events()["MET_ptt"]


def test_a_member_off_the_source_narrows_nothing() -> None:
    session = Session(AwkwardBackend(behavior=BEHAVIOR))
    first = from_awkward("First", EVENTS, session=session)
    second = from_awkward("Second", EVENTS, session=session)
    source = next(nid for nid in session.source_ids() if session.source_name(nid) == "First")
    assert graphed.read_columns([first.MET_pt, second.Muon_pt], source) == ("MET_pt",)
    assert graphed.read_columns(second.Muon_pt, source) is None


@pytest.mark.parametrize("error", [ValueError, AttributeError])
def test_a_behaviors_own_failure_keeps_its_reason(error: type[Exception]) -> None:
    class Failing(ak.Array):  # type: ignore[misc]
        def __getattr__(self, name: str) -> Any:
            if name == "broken":
                raise error("corrections are missing")
            return super().__getattr__(name)

    behavior: dict[Any, Any] = {("*", "Muon"): Failing}
    session = Session(AwkwardBackend(behavior=behavior))
    ev = from_awkward("Events", EVENTS, session=session)
    with pytest.raises(MissingFieldError, match="corrections are missing"):
        ev.Muon.broken  # noqa: B018


def test_a_scalar_says_what_is_missing_not_what_the_backend_lacks() -> None:
    ev = _events()
    with pytest.raises(MissingFieldError, match="no field named 'ptt'") as caught:
        graphed.awkward.gak.sum(ev.MET_pt, axis=None).ptt  # noqa: B018
    assert "TypeTracerArray" not in str(caught.value)


@pytest.mark.parametrize(("typo", "refused"), [("layoutt", "layout"), ("maskk", "mask")])
def test_a_suggestion_never_names_what_a_deferred_array_refuses(typo: str, refused: str) -> None:
    ev = _events()
    with pytest.raises(MissingFieldError) as caught:
        getattr(ev.MET_pt, typo)
    assert "did you mean" not in str(caught.value), refused
    with pytest.raises(AttributeError, match="cannot answer"):
        getattr(ev.MET_pt, refused)


def test_a_field_named_like_a_refused_descriptor_is_still_suggested() -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("Events", ak.Array({"mask": [1.0]}), session=session)
    with pytest.raises(MissingFieldError, match=r"did you mean 'mask'"):
        ev.maskk  # noqa: B018
