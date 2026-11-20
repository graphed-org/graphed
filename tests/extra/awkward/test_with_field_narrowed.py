"""``ak.with_field`` on a typetracer runs over only the fields it changes (#286)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, _ops, from_awkward, gak
from graphed.awkward.projection import project_buffers_many
from graphed.systematics.varied import universes_of

RNG = np.random.default_rng(0)
COUNTS = RNG.integers(0, 6, 40)
N = int(COUNTS.sum())
JETS = ak.with_name(
    ak.unflatten(
        ak.zip(
            {"pt": RNG.uniform(0, 100, N), "eta": RNG.normal(0, 2, N), "phi": RNG.normal(0, 1, N)}
        ),
        COUNTS,
    ),
    "Jet",
)
DATA = ak.Array({"jets": JETS})


def _selected(ev: ak.Array) -> ak.Array:
    return ev.jets[ev.jets.pt > 20]


def _optional(ev: ak.Array) -> ak.Array:
    sel = _selected(ev)
    srt = sel[ak.argsort(sel.pt)]
    return ak.mask(srt, srt.pt > 30)


SHAPES: dict[str, Callable[[ak.Array], ak.Array]] = {
    "flat record": lambda ev: ak.flatten(ev.jets),
    "list of record": lambda ev: ev.jets,
    "list array of record": lambda ev: ev.jets[ak.num(ev.jets) > 1],
    "indexed record": _selected,
    "option of record": _optional,
    "regular record": lambda ev: ak.to_regular(ev.jets[:, :0], axis=1),
    "option of list of record": lambda ev: ak.pad_none(ev.jets, 1, axis=0),
    "option of list of indexed record": lambda ev: ak.pad_none(_selected(ev), 1, axis=0),
}
STEPS = [("pt", "f0"), ("eta", "f1"), ("f0", "f2"), ("phi", "pt"), ("f1", "f3"), ("f3", "eta")]


def _grown(shape: str, with_field: Callable[..., Any]) -> tuple[Any, list[str], list[str]]:
    tt, report = ak.typetracer.typetracer_with_report(DATA.layout.form_with_key())
    rec = SHAPES[shape](ak.Array(tt))
    for src, name in STEPS:
        rec = with_field(rec, rec[src] * 1.0, name)
    at_call = sorted(report.data_touched)
    ak.typetracer.touch_data(rec.f2 + rec.pt)
    return rec, at_call, sorted(report.data_touched)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_form_and_touched_buffers_are_awkwards(shape: str) -> None:
    mine, at_call, after = _grown(shape, _ops._with_field)
    theirs, at_call_theirs, after_theirs = _grown(shape, ak.with_field)
    assert mine.layout.form == theirs.layout.form
    assert mine.type == theirs.type  # parameters, the record's name included
    assert (at_call, after) == (at_call_theirs, after_theirs)


@pytest.fixture
def widths(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Record the width of each record ``ak.with_field`` is handed."""
    seen: list[int] = []
    real = ak.with_field

    def counting(base: Any, what: Any, where: Any = None, **kw: Any) -> Any:
        layout = base.layout
        while not layout.is_record:
            layout = layout.content
        seen.append(len(layout.fields))
        return real(base, what, where=where, **kw)

    monkeypatch.setattr(ak, "with_field", counting)
    yield seen


@pytest.mark.parametrize("shape", SHAPES)
def test_awkward_sees_only_the_fields_a_call_changes(shape: str, widths: list[int]) -> None:
    tt = ak.typetracer.typetracer_with_report(DATA.layout.form_with_key())[0]
    rec = SHAPES[shape](ak.Array(tt))
    for i in range(12):
        rec = _ops._with_field(rec, rec.pt * 1.0, f"f{i}")
    assert len(rec.fields) == 15
    # the first call may re-index each field of a fresh indexed record; after it, only the
    # previous call's new field and this one's remain for awkward to see
    assert max(widths[1:]) <= 1


def test_a_tuple_record_falls_back_to_awkwards_own_call(widths: list[int]) -> None:
    tt = ak.Array(ak.zip((DATA.jets.pt, DATA.jets.eta)).layout.to_typetracer(forget_length=True))
    out = _ops._with_field(tt, tt["0"], "x")
    assert widths == [2] and out.fields == ["0", "1", "x"]
    assert out.layout.form == ak.with_field(tt, tt["0"], where="x").layout.form


def _record(session: Session, width: int) -> list[graphed.Array]:
    ev = from_awkward("ev", DATA, session=session)
    sel = ev.jets[ev.jets.pt > 20]
    pt = graphed.vary(sel.pt, "s", points={"up": sel.pt * 1.1})
    sel = gak.with_field(sel, pt, "shifted")
    for i in range(width):
        sel = gak.with_field(sel, sel.pt, f"f{i}")
    return [sel.f0 + sel.shifted]


def test_plans_and_projections_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def plan() -> tuple[bytes, list[Any], Any]:
        session = Session(AwkwardBackend())
        outs = list(universes_of(_record(session, 12)[0]))
        ir = bytes(graphed.compile_ir(session, *outs).ir)
        forms = [session.form(o).describe() for o in outs]
        return ir, forms, project_buffers_many(outs)

    fast = plan()
    monkeypatch.setattr(_ops, "_narrowed_with_field", lambda *args: None)
    assert plan() == fast
