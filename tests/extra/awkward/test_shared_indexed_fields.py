"""Recorded forms share indexed sub-layouts over one content (#282)."""

from __future__ import annotations

import gc
from typing import Any, cast

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed.awkward import AwkwardBackend, _ops, from_awkward
from graphed.awkward import backend as awkward_backend
from graphed.awkward.backend import AwkwardForm
from graphed.awkward.projection import project_buffers_many


def _grown(width: int, *, option: bool) -> tuple[graphed.Session, Any]:
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 6, 50)
    n = int(counts.sum())
    jets = ak.unflatten(ak.zip({"pt": rng.uniform(0, 100, n), "eta": rng.normal(0, 2, n)}), counts)
    session = graphed.Session(AwkwardBackend())
    events = from_awkward("ev", ak.Array({"jets": jets}), session=session)
    sel = events.jets[events.jets.pt > 20]
    if option:  # a sort then ak.mask leaves `var * ?record`, whose fields with_field re-wraps
        srt = sel[ak.argsort(sel.pt, ascending=False)]
        sel = ak.mask(srt, srt.pt > 30)
    for i in range(width):
        sel[f"f{i}"] = sel.pt * (i + 1)
    return session, sel


def _distinct_layouts(session: graphed.Session) -> int:
    seen: set[int] = set()
    forms = [cast("AwkwardForm", form).tt for form in session._forms.values()]
    stack = [tt.layout for tt in forms if isinstance(tt, ak.Array)]
    while stack:
        layout = stack.pop()
        if id(layout) in seen:
            continue
        seen.add(id(layout))
        if layout.is_record or layout.is_union:
            stack.extend(layout.contents)
        elif not (layout.is_numpy or layout.is_unknown):
            stack.append(layout.content)
    return len(seen)


def _unshared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(awkward_backend, "_sharing", lambda result: result)


@pytest.mark.parametrize("option", [True, False], ids=["option-record", "record"])
def test_sharing_changes_no_form_and_no_projection(
    option: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, sel = _grown(20, option=option)
    shared = (
        cast("AwkwardForm", session.form(sel)).tt.layout.form,
        repr(project_buffers_many([sel.f7 + sel.eta])),
    )
    _unshared(monkeypatch)
    session, sel = _grown(20, option=option)
    unshared = (
        cast("AwkwardForm", session.form(sel)).tt.layout.form,
        repr(project_buffers_many([sel.f7 + sel.eta])),
    )
    assert shared == unshared


@pytest.mark.parametrize("sharing", [True, False], ids=["shared", "unshared"])
def test_retained_layouts_grow_linearly_in_fields(
    sharing: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # awkward's own with_field, which re-wraps every field (a narrowed call re-wraps none)
    monkeypatch.setattr(_ops, "_narrowed_with_field", lambda *args: None)
    if not sharing:
        _unshared(monkeypatch)
    narrow = _distinct_layouts(_grown(40, option=True)[0])
    wide = _distinct_layouts(_grown(80, option=True)[0])
    if sharing:
        assert wide < 2.5 * narrow
    else:  # each with_field re-wraps every field: W²/2 layouts
        assert wide > 3 * narrow


def _indexed_typetracer(**parameters: Any) -> Any:
    real = ak.contents.IndexedArray(
        ak.index.Index64(np.array([1, 0])), ak.contents.NumpyArray(np.array([1.0, 2.0]))
    )
    return real.to_typetracer(forget_length=True).copy(parameters=parameters or None)


def test_an_indexed_layout_over_the_same_content_is_swapped() -> None:
    first = awkward_backend._shared(_indexed_typetracer())
    second = first.copy(index=ak.index.Index64(first.index.data))
    assert second is not first
    assert awkward_backend._shared(second) is first


def test_an_indexed_layout_with_parameters_is_not_swapped() -> None:
    first = awkward_backend._shared(_indexed_typetracer(tag="a"))
    second = first.copy(index=ak.index.Index64(first.index.data))
    assert awkward_backend._shared(second) is second


def test_a_report_bound_index_is_not_swapped() -> None:
    form = _indexed_typetracer().form
    first, _ = ak.typetracer.typetracer_with_report(form, highlevel=False)
    first = awkward_backend._shared(first)
    second = first.copy(index=ak.index.Index64(first.index.data))
    assert awkward_backend._shared(second) is second


def test_a_known_length_index_is_not_swapped() -> None:
    real = ak.contents.IndexedArray(
        ak.index.Index64(np.array([1, 0])), ak.contents.NumpyArray(np.array([1.0, 2.0]))
    )
    first = awkward_backend._shared(real.to_typetracer())
    second = first.copy(index=ak.index.Index64(first.index.data))
    assert awkward_backend._shared(second) is second


def test_entries_die_with_the_session() -> None:
    gc.collect()
    before = len(awkward_backend._CANONICAL), len(awkward_backend._KEPT)
    session, sel = _grown(10, option=True)
    assert len(awkward_backend._CANONICAL) > before[0]
    del session, sel
    gc.collect()
    assert (len(awkward_backend._CANONICAL), len(awkward_backend._KEPT)) == before
