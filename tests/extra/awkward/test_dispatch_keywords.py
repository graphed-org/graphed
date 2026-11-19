"""ak's ``behavior=``, ``attrs=`` and ``highlevel=`` reach a deferred call (#256)."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import awkward as ak
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.awkward import functions as gak_module
from graphed.errors import GraphedTypeError

KEYWORDS = {"behavior": None, "attrs": None, "highlevel": True}
EVENTS = ak.Array(
    {
        "met": [1.0, 2.0, 3.0],
        "jet": [[{"pt": 30.0, "eta": 0.1}], [], [{"pt": 10.0, "eta": -1.0}]],
    }
)


def _mirrored() -> list[str]:
    """Every dispatched gak function whose ak function takes one of the three keywords."""
    return sorted(
        name
        for name in gak_module.GAK_DISPOSITIONS
        if callable(getattr(ak, name, None))
        and KEYWORDS.keys() & inspect.signature(getattr(ak, name)).parameters.keys()
    )


def _operands(session: Session) -> dict[str, Any]:
    ev = from_awkward("ev", EVENTS, session=session)
    return {
        "flat": ev.met,
        "jagged": ev.jet.pt,
        "record": ev.jet,
        "mask": ev.jet.pt > 20.0,
        "option": gak.firsts(ev.jet.pt),
    }


def _fill(value: object, owned: Mapping[str, Any]) -> Any:
    if isinstance(value, gak_module.GakSlot):
        return owned[value.kind]
    if isinstance(value, Mapping):
        return {key: _fill(item, owned) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return type(value)(_fill(item, owned) for item in value)  # type: ignore[call-arg]
    return value


def test_the_repro_zips_with_behavior_none_as_eager_does() -> None:
    session = Session(AwkwardBackend())
    src = from_awkward("e", ak.zip({"x": [1.0, 2.0], "y": [3.0, 4.0]}), session=session)
    out = ak.zip({"x": src.x, "y": src.y}, with_name="V", behavior=None)
    assert str(session.form(out).describe()) == "## * V[x: float64, y: float64]"
    eager = ak.zip({"x": [1.0, 2.0], "y": [3.0, 4.0]}, with_name="V", behavior=None)
    assert ak.to_list(session.materialize(out)) == ak.to_list(eager)


def test_every_mirrored_function_states_the_keywords_its_ak_function_takes() -> None:
    names = _mirrored()
    assert len(names) >= 56
    for name in names:
        ak_params = inspect.signature(getattr(ak, name)).parameters
        gak_params = inspect.signature(getattr(gak, name)).parameters
        for keyword, default in KEYWORDS.items():
            if keyword in ak_params:
                assert gak_params[keyword].default == default, f"gak.{name} lacks {keyword}="


@pytest.mark.parametrize("name", _mirrored())
def test_a_mirrored_function_accepts_the_keywords_at_their_defaults(name: str) -> None:
    session = Session(AwkwardBackend())
    owned = _operands(session)
    fixture = gak_module.GAK_ARG_FIXTURES[name]
    args, kwargs = _fill(fixture.args, owned), _fill(fixture.kwargs, owned)
    ak_params = inspect.signature(getattr(ak, name)).parameters
    defaults = {k: v for k, v in KEYWORDS.items() if k in ak_params}
    assert getattr(gak, name)(*args, **kwargs, **defaults) is not None


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("highlevel", False),
        ("attrs", {"a": 1}),
        ("right_broadcast", True),
        ("optiontype_outside_record", True),
        ("behavior", {"V": type("V", (ak.Record,), {})}),
    ],
)
def test_a_non_default_keyword_graphed_cannot_honour_is_refused_by_name(
    keyword: str, value: Any
) -> None:
    session = Session(AwkwardBackend())
    ev = from_awkward("ev", EVENTS, session=session)
    calls: list[Callable[[], Any]] = [
        lambda: ak.zip({"a": ev.met, "b": ev.met}, **{keyword: value}),
        lambda: gak.zip({"a": ev.met, "b": ev.met}, **{keyword: value}),
    ]
    for call in calls:
        with pytest.raises(GraphedTypeError, match=f"{keyword}="):
            call()


@pytest.fixture
def unregistered_vector() -> Any:
    vector = pytest.importorskip("vector")
    if vector._awkward_registered:
        pytest.skip("vector is registered with awkward in this process")
    return vector


def _vector_case(vector: Any, session: Session) -> tuple[Any, ak.Array]:
    data = ak.zip({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    src = from_awkward("e", data, session=session)
    return vector.zip({"x": src.x, "y": src.y}), vector.zip({"x": data.x, "y": data.y})


def test_vector_zip_runs_deferred_with_the_backend_holding_its_behavior(
    unregistered_vector: Any,
) -> None:
    behavior = unregistered_vector.backends.awkward.behavior
    session = Session(AwkwardBackend(behavior))
    out, eager = _vector_case(unregistered_vector, session)
    value: Any = session.materialize(out)
    assert (str(ak.type(value)), ak.to_list(value)) == (str(ak.type(eager)), ak.to_list(eager))
    assert ak.to_list(value.rho) == ak.to_list(eager.rho)


def test_vector_zip_without_its_behavior_anywhere_is_refused_by_name(
    unregistered_vector: Any,
) -> None:
    session = Session(AwkwardBackend())
    with pytest.raises(GraphedTypeError, match=r"behavior=.*register"):
        _vector_case(unregistered_vector, session)


def test_vector_zip_runs_deferred_once_vector_is_registered(
    unregistered_vector: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ak, "behavior", dict(ak.behavior))
    monkeypatch.setattr(unregistered_vector, "_awkward_registered", False)
    unregistered_vector.register_awkward()
    session = Session(AwkwardBackend())
    out, eager = _vector_case(unregistered_vector, session)
    value: Any = session.materialize(out)
    assert (str(ak.type(value)), ak.to_list(value.rho)) == (
        str(ak.type(eager)),
        ak.to_list(eager.rho),
    )
