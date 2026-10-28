"""Every array-less reader takes ``session=`` as an optional keyword; the positional form warns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import graphed.numpy as gn
from graphed import Session, current_session, set_default_backend, set_session
from graphed.numpy import NumpyBackend, NumpyForm
from graphed.parquet import deferred_source

VALUES = np.arange(3.0)
FORM = NumpyForm(np.dtype("float64"), shape=(None,))


def _loader() -> object:
    return VALUES


# each pair records one source both ways into the Session it is handed
PAIRS: dict[str, tuple[Callable[[Session], Any], Callable[[Session], Any]]] = {
    "from_array": (
        lambda s: gn.from_array(s, "x", VALUES),
        lambda s: gn.from_array("x", VALUES, session=s),
    ),
    "zeros": (lambda s: gn.zeros(s, (3,)), lambda s: gn.zeros((3,), session=s)),
    "zeros-dtype": (
        lambda s: gn.zeros(s, (3,), float),
        lambda s: gn.zeros((3,), float, session=s),
    ),
    "arange": (lambda s: gn.arange(s, 0, 5, 1), lambda s: gn.arange(0, 5, 1, session=s)),
    "deferred_source": (
        lambda s: deferred_source(s, "n", paths=["p"], form=FORM, loader=_loader),
        lambda s: deferred_source("n", paths=["p"], form=FORM, loader=_loader, session=s),
    ),
}


@pytest.mark.parametrize("pair", PAIRS.values(), ids=PAIRS.keys())
def test_the_positional_form_warns_and_records_the_keyword_forms_node(
    pair: tuple[Callable[[Session], Any], Callable[[Session], Any]],
) -> None:
    legacy, keyword = pair
    s = Session(NumpyBackend())
    with pytest.warns(DeprecationWarning, match=r"\(session, \.\.\.\) is deprecated"):
        old = legacy(s)
    new = keyword(s)
    assert old.session is new.session is s
    assert old.node_id == new.node_id


def test_from_awkward_positional_and_keyword_bind_one_slot() -> None:
    ak = pytest.importorskip("awkward")
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415

    s = Session(AwkwardBackend())
    events = ak.Array({"x": [1.0, 2.0]})
    with pytest.warns(DeprecationWarning, match=r"from_awkward\(session, \.\.\.\)"):
        old = from_awkward(s, "events", events)
    assert from_awkward("events", events, session=s).node_id == old.node_id


def test_from_record_positional_and_keyword_record_the_same_graph() -> None:
    a, b = Session(NumpyBackend()), Session(NumpyBackend())
    with pytest.warns(DeprecationWarning, match=r"from_record\(session, \.\.\.\)"):
        old = gn.from_record(a, "r", x=VALUES)
    new = gn.from_record("r", session=b, x=VALUES)
    assert old.node_id == new.node_id
    assert a.serialized_ir(old, optimize=False) == b.serialized_ir(new, optimize=False)


def test_a_reader_lands_in_the_open_scope_else_the_default() -> None:
    default = current_session(NumpyBackend())
    scoped = Session(NumpyBackend())
    with scoped:
        assert gn.from_array("x", VALUES).session is scoped
        assert gn.default_rng(7).normal(3).session is scoped
    assert gn.from_array("x", VALUES).session is default


def test_a_reader_makes_the_default_on_its_own_backend_whatever_the_factory() -> None:
    pytest.importorskip("awkward")
    from graphed.awkward import AwkwardBackend  # noqa: PLC0415

    previous = set_default_backend(AwkwardBackend)
    try:
        made = gn.from_array("x", VALUES).session
    finally:
        set_default_backend(previous)
    assert isinstance(made.backend, NumpyBackend)
    assert made is current_session()


def test_a_reader_refuses_a_session_on_another_backend() -> None:
    ak = pytest.importorskip("awkward")
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415

    set_session(Session(AwkwardBackend()))
    with pytest.raises(
        TypeError, match="records on AwkwardBackend, and this reader records on Num"
    ):
        gn.from_array("x", VALUES)
    with Session(NumpyBackend()), pytest.raises(TypeError, match=r"records on NumpyBackend.*Awk"):
        from_awkward("events", ak.Array({"x": [1.0]}))


def test_a_session_passed_both_ways_is_refused() -> None:
    s = Session(NumpyBackend())
    with pytest.raises(TypeError, match="both positionally and as session="):
        gn.from_array(s, "x", VALUES, session=s)  # type: ignore[call-overload]


def test_session_is_a_reserved_keyword() -> None:
    with pytest.raises(TypeError, match=r"session= must be a graphed\.Session, not ndarray"):
        gn.from_record("r", session=np.arange(3))  # type: ignore[call-overload]
