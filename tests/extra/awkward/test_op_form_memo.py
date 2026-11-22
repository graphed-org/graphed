"""A Session infers an op's form once per backend key: op, parameters and input forms."""

from __future__ import annotations

import gc
import warnings
import weakref
from collections import Counter
from typing import Any, cast

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.awkward.backend as backend
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.awkward._ops import apply
from graphed.errors import GraphedTypeError

UNIVERSES = 8


@pytest.fixture
def inferred(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    counts: Counter[str] = Counter()

    def counted(op: str, *args: Any, **kwargs: Any) -> Any:
        counts[op] += 1
        return apply(op, *args, **kwargs)

    monkeypatch.setattr(backend, "apply", counted)
    return counts


def _universes(session: Session) -> list[Any]:
    ev = from_awkward("ev", ak.Array({"a": [1.0, 2.0, 3.0]}), session=session)
    return [ev.a * (1.0 + i) for i in range(UNIVERSES)]  # one node each, all of one form


def test_universes_of_one_op_infer_its_form_once(inferred: Counter[str]) -> None:
    session = Session(AwkwardBackend())
    shifted = _universes(session)
    added = [x + 1.0 for x in shifted]
    assert len({a.node_id for a in added}) == UNIVERSES
    assert inferred["add"] == 1
    assert len({id(session.form(a)) for a in added}) == 1  # every node holds the one form
    materialized = [graphed.materialize(a).tolist() for a in added[:2]]
    assert materialized == [[2.0, 3.0, 4.0], [3.0, 5.0, 7.0]]


def test_a_differing_parameter_infers_again(inferred: Counter[str]) -> None:
    x, y = _universes(Session(AwkwardBackend()))[:2]
    x + 1.0, y + 1.0
    y + 2.0
    assert inferred["add"] == 2


def test_a_differing_input_form_infers_again(inferred: Counter[str]) -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", ak.Array({"a": [1.0, 2.0], "n": [1, 2]}), session=session)
    ev.a + 1.0, (ev.a * 2.0) + 1.0
    ev.n + 1.0
    assert inferred["add"] == 2


class _Loud(ak.Record):  # type: ignore[misc]
    pass


def _loud_add(a: Any, b: Any) -> Any:
    warnings.warn("loud add", UserWarning, stacklevel=2)
    return ak.zip({"v": a.v + b.v}, with_name="Loud")


class _LoudArray(ak.Array):  # type: ignore[misc]
    def scaled(self, k: float) -> Any:
        return self.v * k


def test_a_behavior_registered_between_two_recordings_is_seen(
    inferred: Counter[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = Session(AwkwardBackend())
    one, other = (  # two universes of one form
        from_awkward(name, ak.Array({"v": [1.0, 2.0]}), session=session) for name in ("1", "2")
    )
    before = cast("AwkwardForm", session.form(gak.with_name(one, "Late")))
    monkeypatch.setitem(ak.behavior, ("*", "Late"), _LoudArray)
    after = cast("AwkwardForm", session.form(gak.with_name(other, "Late")))
    assert type(before.tt) is ak.Array
    assert type(after.tt) is _LoudArray
    assert inferred["ak.with_name"] == 2


def test_an_inference_warning_is_emitted_once_per_key(inferred: Counter[str]) -> None:
    behavior: dict[Any, Any] = {"Loud": _Loud, (np.add, "Loud", "Loud"): _loud_add}
    session = Session(AwkwardBackend(behavior))
    caught = []
    for name in ("one", "other"):  # two universes of one form
        ev = from_awkward(name, ak.zip({"v": [1.0, 2.0]}, with_name="Loud"), session=session)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            ev + ev
        caught.append([str(w.message) for w in seen])
    assert caught == [["loud add"], []]
    assert inferred["add"] == 1


def test_a_warning_raised_as_an_error_raises_at_every_site() -> None:
    behavior: dict[Any, Any] = {"Loud": _Loud, (np.add, "Loud", "Loud"): _loud_add}
    session = Session(AwkwardBackend(behavior))
    for name in ("one", "other"):
        ev = from_awkward(name, ak.zip({"v": [1.0, 2.0]}, with_name="Loud"), session=session)
        with warnings.catch_warnings(), pytest.raises(GraphedTypeError, match="loud add"):
            warnings.simplefilter("error")
            ev + ev


def test_an_inference_that_raises_raises_at_every_site(inferred: Counter[str]) -> None:
    session = Session(AwkwardBackend())
    for name in ("one", "other"):
        ev = from_awkward(name, ak.Array({"s": ["x", "y"]}), session=session)
        with pytest.raises(GraphedTypeError):
            ev.s + 1.0
    assert inferred["add"] == 2


def test_a_wide_record_is_keyed_by_its_inference_not_its_structure(inferred: Counter[str]) -> None:
    session = Session(AwkwardBackend())
    wide = {f"f{i}": [1.0, 2.0] for i in range(backend._WIDE)}
    widened = []
    for name in ("one", "other"):  # two universes of one wide source form
        ev = from_awkward(name, ak.Array(wide), session=session)
        widened.append(gak.with_field(ev, ev.f0 * 2.0, "g"))
    picked = [w.g for w in widened]
    assert inferred["ak.with_field"] == 1
    assert inferred["field"] == 2  # f0 of the source, then g of the one widened form
    form = cast("AwkwardForm", session.form(widened[0]))
    assert form is session.form(widened[1])
    assert form.key == id(form)
    assert [graphed.materialize(p).tolist() for p in picked] == [[2.0, 4.0], [2.0, 4.0]]


def test_a_parameter_without_a_plain_key_has_no_memo_key() -> None:
    keyed = AwkwardBackend()
    session = Session(keyed)
    form = cast("AwkwardForm", session.form(_universes(session)[0]))
    assert keyed.op_key("add", [form], {"scalar": 1.0}) is not None
    assert keyed.op_key("add", [form], {"scalar": object()}) is None


def _warns_then_raises(a: Any, b: Any) -> Any:
    warnings.warn("before the raise", UserWarning, stacklevel=2)
    raise ValueError("loud failure")


def test_an_inference_that_warns_then_raises_shows_its_warning() -> None:
    behavior: dict[Any, Any] = {"Loud": _Loud, (np.add, "Loud", "Loud"): _warns_then_raises}
    session = Session(AwkwardBackend(behavior))
    ev = from_awkward("ev", ak.zip({"v": [1.0]}, with_name="Loud"), session=session)
    with pytest.warns(UserWarning, match="before the raise"), pytest.raises(GraphedTypeError):
        ev + ev


def test_universes_of_one_method_call_run_it_once(inferred: Counter[str]) -> None:
    behavior: dict[Any, Any] = {"Loud": _Loud, ("*", "Loud"): _LoudArray}
    session = Session(AwkwardBackend(behavior))
    scaled = []
    for name in ("one", "other", "third", "fourth"):  # universes of one form
        ev = from_awkward(name, ak.zip({"v": [1.0, 2.0]}, with_name="Loud"), session=session)
        scaled.append(ev.scaled(2.0))
    assert inferred["method"] == 2  # the call's outputs, then its form, at the first site alone
    assert graphed.materialize(scaled[-1]).tolist() == [2.0, 4.0]


def test_the_memo_dies_with_its_session() -> None:
    kept = AwkwardBackend()
    session = Session(kept)
    ev = from_awkward("ev", ak.Array({"a": [1.0, 2.0]}), session=session)
    form = weakref.ref(session.form(ev.a + 1.0))
    del session, ev
    gc.collect()
    assert form() is None
    assert isinstance(kept, AwkwardBackend)
