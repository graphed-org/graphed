"""m60 integ-m60-O — two callables never share a node by accident.

An opaque callable's identity travels in the `fn` param (every backend derives the payload's
content hash from it), so two distinct objects that DERIVE the same name intern to one node and
the second one's result is the first one's. Identity becomes a per-Session first-seen ordinal on
that derived name; `name=` stays the caller's own declaration and is never ordinaled.
"""

from __future__ import annotations

import pickle
from typing import Any

import m60_lib
import m60_libx
import numpy as np
import pytest
from m60_seams import (
    DATUM,
    ToyBackend,
    fn_params,
    nodes_of,
    numpy_session,
    plus_one,
    scaler,
    times_hundred,
    toy_session,
)

import graphed
import graphed.numpy as gnp
from graphed import Array, apply, compile_ir, evaluate_ir
from graphed.execute import external_key

#: three ways two DISTINCT callable objects derive one name: no name at all, one factory's code,
#: and two modules that spell the same function name
PAIRS: dict[str, tuple[Any, Any]] = {
    "lambdas": (lambda value: value + 1, lambda value: value * 100),
    "closures": (scaler(1), scaler(100)),
    "modules": (m60_lib.q, m60_libx.q),
}


def _record(surface: str, array: Array, fn: Any) -> Array:
    return array.map(fn) if surface == "map" else apply(fn, array)


@pytest.mark.parametrize("surface", ["map", "apply"])
@pytest.mark.parametrize("pair", list(PAIRS))
def test_two_distinct_callables_are_two_nodes_that_evaluate_to_their_own_results(
    pair: str, surface: str
) -> None:
    first, second = PAIRS[pair]
    assert first.__name__ == second.__name__ and first is not second
    session, x = toy_session()

    left, right = _record(surface, x, first), _record(surface, x, second)

    assert left.node_id != right.node_id
    assert session.node_count() == 3
    assert (session.materialize(left), session.materialize(right)) == (first(DATUM), second(DATUM))
    assert len(set(fn_params(session, left, right))) == 2


@pytest.mark.parametrize("pair", list(PAIRS))
def test_two_distinct_callables_are_two_gufunc_nodes(pair: str) -> None:
    first, second = PAIRS[pair]
    session, x = numpy_session()

    left = gnp.apply_gufunc(first, "()->()", x, output_dtype="float64")
    right = gnp.apply_gufunc(second, "()->()", x, output_dtype="float64")

    assert left.node_id != right.node_id
    assert session.node_count() == 3
    assert np.array_equal(session.materialize(left), first(np.asarray(session.materialize(x))))
    assert np.array_equal(session.materialize(right), second(np.asarray(session.materialize(x))))


def test_the_second_object_of_a_derived_name_records_that_name_with_an_ordinal() -> None:
    session, x = toy_session()

    first, second = x.map(m60_lib.q), x.map(m60_libx.q)

    assert fn_params(session, first, second) == ["q", "q#1"]


def test_the_durable_path_ships_each_callable_its_own_evaluator() -> None:
    """`compile_ir` carries no callables (an opaque payload is a preservation risk, §A.3.1): the
    worker resolves them by payload key, so two colliding callables share ONE evaluator there."""
    session, x = toy_session()
    first, second = x.map(m60_lib.q), x.map(m60_libx.q)

    shipped = pickle.loads(pickle.dumps(compile_ir(session, first, second, optimize=False)))
    externals = [node for node in nodes_of(session, first, second) if node["kind"] == "external"]
    keys = [external_key(node) for node in externals]

    assert len(set(keys)) == 2
    bound = dict(zip(keys, (m60_lib.q, m60_libx.q), strict=True))
    assert evaluate_ir(shipped, ToyBackend(), {"x": DATUM}, externals=bound) == [
        m60_lib.q(DATUM),
        m60_libx.q(DATUM),
    ]


def test_one_callable_object_recorded_twice_is_one_node_and_apply_interns_with_map() -> None:
    session, x = toy_session()

    mapped, again = x.map(plus_one), x.map(plus_one)
    applied = apply(plus_one, x)

    assert mapped.node_id == again.node_id == applied.node_id
    assert session.node_count() == 2
    assert session.materialize(mapped) == plus_one(DATUM)


def test_a_program_without_name_collisions_records_the_bare_names_and_the_same_ir_twice() -> None:
    built = []
    for _ in range(2):
        session, x = toy_session()
        first, second = x.map(plus_one), x.map(times_hundred)
        built.append((session.serialized_ir(first, second, optimize=False), fn_params(session, first, second)))

    assert built[0][1] == built[1][1] == ["plus_one", "times_hundred"]
    assert built[0][0] == built[1][0]


def test_two_builds_of_a_colliding_program_agree_byte_for_byte() -> None:
    built = set()
    for _ in range(2):
        session, x = toy_session()
        first, second = x.map(m60_lib.q), x.map(m60_libx.q)
        built.add(session.serialized_ir(first, second, optimize=False))

    assert len(built) == 1


def test_an_explicit_name_is_the_callers_identity_declaration() -> None:
    session, x = toy_session()

    shared = x.map(lambda value: value + 1, name="shared")
    also_shared = x.map(lambda value: value * 100, name="shared")
    other = x.map(lambda value: value + 1, name="other")

    assert shared.node_id == also_shared.node_id != other.node_id
    assert fn_params(session, shared, other) == ["shared", "other"]
    for verb in (Array.map, graphed.apply, gnp.apply_gufunc):
        assert verb.__doc__ is not None and "name=" in verb.__doc__
