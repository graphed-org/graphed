"""#143: one name= answers for one function."""

from __future__ import annotations

import sys
import threading
import warnings
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
from graphed import GraphedError, GraphedTypeError, GraphedWarning
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.numpy import NumpyBackend, apply_gufunc, from_array
from graphed.session import Session


def here() -> int:
    return sys._getframe(1).f_lineno


def events() -> tuple[Session, Any]:
    session = Session(AwkwardBackend())
    data = ak.Array({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    return session, from_awkward("ev", data, session=session)


def scaled(by: Any) -> Any:
    return lambda values: values * by


def double(values: Any) -> Any:
    return values * 2


def thousand(values: Any) -> Any:
    return values * 1000


def test_a_different_function_is_refused_naming_the_first_and_its_line() -> None:
    _, ev = events()
    first_line = here() + 1
    ev.x.map(double, name="scale")
    with pytest.raises(GraphedError) as caught:
        second_line = here() + 1
        ev.x.map(thousand, name="scale")
    assert f"name='scale' at {__file__}:{second_line}" in str(caught.value)
    assert f"already used by double ({__file__}:{first_line})" in str(caught.value)


def test_the_refusal_covers_a_name_reused_over_other_inputs() -> None:
    _, ev = events()
    ev.x.map(double, name="scale")
    with pytest.raises(GraphedError, match="name='scale'"):
        ev.y.map(thousand, name="scale")


def test_a_refused_call_records_nothing() -> None:
    session, ev = events()
    ev.x.map(double, name="scale")
    before = session.node_count()
    with pytest.raises(GraphedError):
        ev.x.map(thousand, name="scale")
    assert session.node_count() == before


def test_one_function_over_two_inputs_answers_for_each() -> None:
    _, ev = events()
    a, b = ev.x.map(double, name="scale"), (ev.y * 10).map(double, name="scale")
    assert [ak.to_list(v) for v in graphed.materialize([a, b])] == [
        [2.0, 4.0, 6.0],
        [20.0, 40.0, 60.0],
    ]


@pytest.mark.parametrize("shift", ["", "\n\n\n"], ids=["same-line", "moved"])
def test_a_rerun_cell_gives_the_same_node(shift: str) -> None:
    _, ev = events()
    namespace: dict[str, Any] = {}
    source = f"{shift}f = lambda values: values * 2\n"
    exec(compile("f = lambda values: values * 2\n", "<cell 1>", "exec"), namespace)
    first = ev.x.map(namespace["f"], name="scale")
    exec(compile(source, "<cell 2>", "exec"), namespace)
    assert ev.x.map(namespace["f"], name="scale").node_id == first.node_id


def test_closures_compare_by_what_they_captured() -> None:
    _, ev = events()
    first = ev.x.map(scaled(2), name="scale")
    assert ev.x.map(scaled(2), name="scale").node_id == first.node_id
    with pytest.raises(GraphedError, match="name='scale'"):
        ev.x.map(scaled(1000), name="scale")


def test_captured_arrays_compare_by_content() -> None:
    _, ev = events()
    ev.x.map(scaled(np.array([1.0, 2.0, 3.0])), name="scale")
    ev.x.map(scaled(np.array([1.0, 2.0, 3.0])), name="scale")
    with pytest.raises(GraphedError):
        ev.x.map(scaled(np.array([1.0, 2.0, 4.0])), name="scale")


def test_functions_that_cannot_be_compared_warn_and_the_first_answers() -> None:
    _, ev = events()
    lock = threading.Lock()

    def uses_lock(values: Any) -> Any:
        return values + lock.locked()

    def also_uses_lock(values: Any) -> Any:
        return values * 2 + lock.locked()

    first = ev.x.map(uses_lock, name="scale")
    with pytest.warns(GraphedWarning, match="could not compare also_uses_lock with uses_lock"):
        second = ev.x.map(also_uses_lock, name="scale")
    assert second.node_id == first.node_id


def test_when_they_cannot_be_compared_the_first_answers_over_any_input() -> None:
    _, ev = events()
    lock = threading.Lock()

    def uses_lock(values: Any) -> Any:
        return values + lock.locked()

    def also_uses_lock(values: Any) -> Any:
        return ak.values_astype(values * 2 + lock.locked(), "int64")

    a = ev.x.map(uses_lock, name="scale")
    with pytest.warns(GraphedWarning, match="the first one answers"):
        b = ev.y.map(also_uses_lock, name="scale")
    assert b.node_id != a.node_id
    assert str(b.session.form(b).describe()) == "## * float64"
    assert ak.to_list(graphed.materialize(b)) == [1.0, 2.0, 3.0]
    assert [ak.to_list(v) for v in graphed.materialize([a, b])] == [[1.0, 2.0, 3.0]] * 2


def test_a_call_the_backend_refuses_does_not_take_the_name() -> None:
    session = Session(NumpyBackend())
    flat = from_array("flat", np.arange(4.0).reshape(2, 2), session=session)
    square = from_array("square", np.arange(8.0).reshape(2, 2, 2), session=session)
    with pytest.raises(GraphedTypeError):
        apply_gufunc(
            lambda a: a.sum(axis=(-2, -1)), "(i,j)->()", flat, name="g", output_dtype="f8"
        )
    fixed = apply_gufunc(
        lambda a: a.max(axis=(-2, -1)), "(i,j)->()", square, name="g", output_dtype="f8"
    )
    assert np.asarray(session.materialize(fixed)).tolist() == [3.0, 7.0]


class Unpicklable:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def double(self, values: Any) -> Any:
        return values * 2


def test_a_bound_method_reread_from_its_owner_is_one_call() -> None:
    _, ev = events()
    owner = Unpicklable()
    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        assert ev.x.map(owner.double).node_id == ev.x.map(owner.double).node_id
        assert ev.x.map(owner.double, name="d").node_id == ev.x.map(owner.double, name="d").node_id


def test_the_same_object_needs_no_comparison() -> None:
    _, ev = events()
    lock = threading.Lock()

    def uses_lock(values: Any) -> Any:
        return values + lock.locked()

    with warnings.catch_warnings():
        warnings.simplefilter("error", GraphedWarning)
        assert (
            ev.x.map(uses_lock, name="scale").node_id == ev.x.map(uses_lock, name="scale").node_id
        )


def test_a_declared_type_keeps_a_name_apart() -> None:
    _, ev = events()
    a = ev.x.map(double, name="scale", output_type="float64")
    b = ev.x.map(thousand, name="scale", output_type="float32")
    assert a.node_id != b.node_id


def test_graphed_apply_is_covered() -> None:
    _, ev = events()
    graphed.apply(lambda a, b: a + b, ev.x, ev.y, name="pair")
    with pytest.raises(GraphedError, match="name='pair'"):
        graphed.apply(lambda a, b: a - b, ev.x, ev.y, name="pair")


def test_apply_gufunc_is_covered() -> None:
    session = Session(NumpyBackend())
    x = from_array("x", np.arange(4.0).reshape(2, 2), session=session)
    apply_gufunc(lambda a: a.sum(axis=-1), "(i)->()", x, name="g", output_dtype="f8")
    with pytest.raises(GraphedError, match="name='g'"):
        apply_gufunc(lambda a: a.max(axis=-1), "(i)->()", x, name="g", output_dtype="f8")


def test_a_derived_name_is_held_too() -> None:
    _, ev = events()
    ev.x.map(double)
    with pytest.raises(GraphedError, match="already used by double"):
        ev.x.map(thousand, name="double")


def test_undeclared_lambdas_are_never_compared() -> None:
    _, ev = events()
    a, b = ev.x.map(lambda v: v * 2), ev.x.map(lambda v: v * 1000)
    assert [ak.to_list(v) for v in graphed.materialize([a, b])] == [
        [2.0, 4.0, 6.0],
        [1000.0, 2000.0, 3000.0],
    ]
