"""m60 integ-m60-O — the two ways one node could still be handed to two callables.

"The same callable, asked for again": `obj.method` is a fresh object with a fresh id per access,
so a callable with an owner is identified by that owner's IDENTITY plus its function — never by
`==`/`hash`, which a class is free to declare of two behaviourally distinct callables.
"A derived name equal to one already handed to a DIFFERENT callable": declared and derived names
share one space, or the next un-named callable deriving a declared name takes its node and result.

Toy-backed and numpy-idiom (the free-threaded frontend job installs no awkward); `m13_toy`'s
backend is reached by bare name through the pytest `pythonpath`, and its sources carry data here,
so each node's IDENTITY and its evaluated VALUE are both observable.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Callable
from typing import Any

import pytest
from m13_toy import ToyBackend, ToyForm

import graphed.core
from graphed import Array, Session, apply

#: the toy source value; every answer below is exact on it
DATUM = 2


class Corr:
    """A corrections object whose BOUND METHOD is the callable an analyst hands to `map`."""

    def __init__(self, k: float) -> None:
        self.k = k

    def scale(self, value: Any) -> Any:
        return value * self.k

    def offset(self, value: Any) -> Any:
        """A second method on the same owner: same `__self__`, a different `__func__`."""
        return value + self.k


class UnhashableOwner:
    """An OWNER that cannot be a dict key at all — its bound method keys on the owner's id."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, k: float) -> None:
        self.k = k

    def scale(self, value: Any) -> Any:
        return value * self.k


@dataclasses.dataclass(frozen=True)
class Calib:
    """An OWNER whose class calls two of these EQUAL: the behaviour field is `compare=False`."""

    label: str
    factor: float = dataclasses.field(compare=False)

    def scale(self, value: Any) -> Any:
        return value * self.factor


class Riders:
    """Two `functools.partialmethod`s on ONE owner.

    Each access builds a `functools.partial` with a COPIED `__self__` and no `__func__` and no
    `__name__`, so the owner alone cannot tell the two apart.
    """

    def _scale(self, value: Any, factor: float) -> Any:
        return value * factor

    doubled = functools.partialmethod(_scale, factor=2.0)
    hundredfold = functools.partialmethod(_scale, factor=100.0)


class Proxy:
    """A transparent proxy over a real method: `__class__`, `__self__` and `__func__` forward.

    `isinstance` consults `__class__`, so this CLAIMS to be a method; its call is its own.
    """

    def __init__(self, method: Any, factor: float) -> None:
        self._method, self.factor = method, factor

    @property  # type: ignore[misc]  # the false claim under test
    def __class__(self) -> Any:
        return self._method.__class__

    @property
    def __self__(self) -> Any:
        return self._method.__self__

    @property
    def __func__(self) -> Any:
        return self._method.__func__

    def __call__(self, value: Any) -> Any:
        return value * self.factor


class Unhashable:
    """A callable that cannot be a dict key at all — the memo must key it on identity instead."""

    __name__ = "unhashable"
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, k: float) -> None:
        self.k = k

    def __call__(self, value: Any) -> Any:
        return value * self.k


@dataclasses.dataclass(frozen=True)
class Weight:
    """A callable whose class calls two of these EQUAL: the behaviour field is `compare=False`."""

    label: str
    factor: float = dataclasses.field(compare=False)

    __name__ = "weight"

    def __call__(self, value: Any) -> Any:
        return value * self.factor


class Trigger:
    """The same trap hand-written: `__eq__`/`__hash__` on a key field, behaviour on another."""

    __name__ = "trigger"

    def __init__(self, label: str, factor: float) -> None:
        self.label, self.factor = label, factor

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Trigger) and self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __call__(self, value: Any) -> Any:
        return value * self.factor


def scale(value: Any) -> Any:
    """A free function spelled like `Corr.scale`, with no `self` in sight."""
    return value * 1000


def q(value: Any) -> Any:
    return value + 1000


def q_closure(factor: float) -> Any:
    """A second distinct callable named `q`: one factory's code, its own object."""

    def q(value: Any) -> Any:
        return value * factor

    return q


def zero(value: Any) -> Any:
    return value * 0.0


def toy_session() -> tuple[Session, Array]:
    session = Session(ToyBackend())
    return session, session.source("x", form=ToyForm("source"), data=DATUM)


def fn_params(session: Session, *outputs: Array) -> list[str]:
    """The `fn` param of every External node reaching `outputs`, in record order."""
    graph = graphed.core.GraphStore.deserialize(session.serialized_ir(*outputs, optimize=False))
    return [str(node["params"]["fn"]) for node in graph.nodes() if node["kind"] == "external"]


# ---- (1) the same callable, asked for again ---------------------------------------------------
def test_one_bound_method_asked_for_again_is_one_node_and_applies_intern_with_it() -> None:
    session, x = toy_session()
    c = Corr(2)
    assert c.scale is not c.scale  # a fresh object per access: an id memo could never hit

    mapped, again, applied = x.map(c.scale), x.map(c.scale), apply(c.scale, x)

    assert mapped.node_id == again.node_id == applied.node_id
    assert session.node_count() == 2
    assert fn_params(session, mapped) == ["scale"]
    assert session.materialize(mapped) == c.scale(DATUM)


def test_a_loop_recording_one_bound_method_stays_two_nodes() -> None:
    session, x = toy_session()
    c = Corr(2)

    for _ in range(6):
        x.map(c.scale)

    assert session.node_count() == 2


def test_a_bound_method_of_an_unhashable_owner_is_one_node() -> None:
    """The merged end where the owner cannot be hashed: only its ID enters the key."""
    session, x = toy_session()
    u = UnhashableOwner(3)
    with pytest.raises(TypeError):
        hash(u)  # the owner cannot be half of a key either

    first, again = x.map(u.scale), x.map(u.scale)

    assert first.node_id == again.node_id
    assert session.node_count() == 2
    assert session.materialize(first) == u.scale(DATUM)


def test_two_method_objects_fabricated_from_one_pair_are_one_node() -> None:
    """`types.MethodType(func, obj)` twice: two objects Python defines as the same call."""
    session, x = toy_session()
    c = Corr(2)
    call, same_call = types.MethodType(Corr.scale, c), types.MethodType(Corr.scale, c)
    assert call is not same_call

    first, again = x.map(call), x.map(same_call)

    assert first.node_id == again.node_id
    assert session.node_count() == 2
    assert session.materialize(first) == c.scale(DATUM)


def test_two_methods_of_one_owner_are_two_nodes() -> None:
    """One `__self__`, two `__func__`s — the owner alone is not the identity."""
    session, x = toy_session()
    c = Corr(2)
    scale_call: Any = c.scale
    offset_call: Any = c.offset
    assert scale_call.__self__ is offset_call.__self__ is c

    scaled, offset = x.map(scale_call), x.map(offset_call)

    assert scaled.node_id != offset.node_id
    assert (session.materialize(scaled), session.materialize(offset)) == (
        c.scale(DATUM),
        c.offset(DATUM),
    )


def test_bound_methods_of_two_equal_owners_are_two_nodes() -> None:
    """The owners are `==` and hash alike, so an owner-as-key memo hands the second the first's
    result — the same collapse `==`-equal CALLABLES offered, one level up."""
    session, x = toy_session()
    c, e = Calib("nominal", 2.0), Calib("nominal", 100.0)
    mine: Any = c.scale
    theirs: Any = e.scale
    assert c == e and hash(c) == hash(e) and mine.__func__ is theirs.__func__

    first, second = x.map(mine), x.map(theirs)

    assert first.node_id != second.node_id
    assert (session.materialize(first), session.materialize(second)) == (
        c.scale(DATUM),
        e.scale(DATUM),
    )


def test_two_partialmethods_on_one_owner_are_two_nodes() -> None:
    """Not `types.MethodType`: a copied `__self__` with no function object behind it."""
    session, x = toy_session()
    r = Riders()
    doubled: Any = r.doubled
    hundredfold: Any = r.hundredfold
    assert doubled.__self__ is hundredfold.__self__ is r
    assert not isinstance(doubled, types.MethodType)
    assert not hasattr(doubled, "__func__") and not hasattr(doubled, "__name__")

    first, second = x.map(doubled), x.map(hundredfold)

    assert first.node_id != second.node_id
    assert (session.materialize(first), session.materialize(second)) == (4.0, 200.0)


def test_a_callable_that_only_claims_to_be_a_method_is_its_own_identity() -> None:
    """Two proxies over ONE held method carry one `(__self__, __func__)` pair and two calls."""
    session, x = toy_session()
    held = types.MethodType(Corr.scale, Corr(2))
    doubled, hundredfold = Proxy(held, 2.0), Proxy(held, 100.0)
    assert isinstance(doubled, types.MethodType) and type(doubled) is not types.MethodType
    assert doubled.__self__ is hundredfold.__self__ and doubled.__func__ is hundredfold.__func__

    first, second = x.map(doubled), x.map(hundredfold)

    assert first.node_id != second.node_id
    assert (session.materialize(first), session.materialize(second)) == (4.0, 200.0)


def test_distinct_bound_methods_and_a_self_free_scale_stay_three_nodes() -> None:
    """The admitted-member control: three callables the class must NOT merge, all named `scale`."""
    session, x = toy_session()
    c, d = Corr(2), Corr(100)
    free_of_self: object = Corr.scale
    assert c.scale != d.scale  # bound-method equality is `__self__` IDENTITY + `__func__`
    assert c.scale != free_of_self and scale.__name__ == Corr.scale.__name__

    mine, theirs, free = x.map(c.scale), x.map(d.scale), x.map(scale)

    assert len({mine.node_id, theirs.node_id, free.node_id}) == 3
    assert fn_params(session, mine, theirs, free) == ["scale", "scale#1", "scale#2"]
    assert session.materialize(mine) == c.scale(DATUM)
    assert session.materialize(theirs) == d.scale(DATUM)


def test_an_unhashable_callable_is_memoed_on_its_identity() -> None:
    session, x = toy_session()
    u, other = Unhashable(3), Unhashable(7)
    with pytest.raises(TypeError):
        hash(u)  # cannot be the memo key itself

    first, again, third = x.map(u), x.map(u), x.map(other)

    assert first.node_id == again.node_id != third.node_id
    assert session.node_count() == 3
    assert fn_params(session, first, third) == ["unhashable", "unhashable#1"]
    assert (session.materialize(first), session.materialize(third)) == (u(DATUM), other(DATUM))


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda k: Weight("nominal", k), id="frozen-dataclass-compare-false"),
        pytest.param(lambda k: Trigger("nominal", k), id="hand-written-eq-and-hash"),
    ],
)
def test_two_callables_their_class_calls_equal_are_still_two_nodes(
    make: Callable[[float], Any],
) -> None:
    """An `==`/`hash` memo would hand the second one the first's node AND the first's result."""
    session, x = toy_session()
    a, b = make(2.0), make(100.0)
    assert a is not b and a == b and hash(a) == hash(b)  # the collapse is available to take

    first, second = x.map(a), x.map(b)

    assert first.node_id != second.node_id
    assert session.node_count() == 3
    assert fn_params(session, first, second) == [a.__name__, f"{a.__name__}#1"]
    assert (session.materialize(first), session.materialize(second)) == (a(DATUM), b(DATUM))


def test_a_method_wrapper_recorded_twice_is_never_a_WRONG_answer() -> None:
    """The accepted ceiling: a method-wrapper is not a `types.MethodType`, so a re-access may mint
    a second node. However many it mints, every one of them materializes to `k * DATUM`."""
    session, x = toy_session()
    k: Any = 3.0
    wrapper, wrapper_again = k.__mul__, k.__mul__  # both held: no id can recycle under us
    assert wrapper is not wrapper_again and not isinstance(wrapper, types.MethodType)

    first, again = x.map(wrapper), x.map(wrapper_again)

    assert {session.materialize(first), session.materialize(again)} == {k * DATUM}


# ---- (3) the `lambda` literal a nameless callable derives -------------------------------------
def test_a_callable_without_a_name_records_the_lambda_literal() -> None:
    session, x = toy_session()
    first, second = functools.partial(scale), functools.partial(q)
    assert not hasattr(first, "__name__")

    left, right = x.map(first), x.map(second)

    assert fn_params(session, left, right) == ["lambda", "lambda#1"]
    assert left.node_id != right.node_id


# ---- (2) a declared name is inside the one name space -----------------------------------------
def test_a_derived_name_never_takes_a_declared_ones_node() -> None:
    session, x = toy_session()

    declared = x.map(zero, name="q")
    derived = x.map(q)

    assert declared.node_id != derived.node_id
    assert fn_params(session, declared, derived) == ["q", "q#1"]
    assert (session.materialize(declared), session.materialize(derived)) == (zero(DATUM), q(DATUM))


def test_a_third_callable_of_a_declared_name_advances_to_the_next_free_ordinal() -> None:
    session, x = toy_session()

    declared = x.map(zero, name="q")
    first, second = x.map(q), x.map(q_closure(3))

    assert len({declared.node_id, first.node_id, second.node_id}) == 3
    assert fn_params(session, declared, first, second) == ["q", "q#1", "q#2"]


def test_a_declared_ordinal_spelling_is_taken_like_any_other_name() -> None:
    """The adversarial member: the declaration pre-empts the ordinal a derivation would reach for."""
    session, x = toy_session()

    decoy = x.map(zero, name="q#1")
    first, second = x.map(q), x.map(q_closure(3))

    assert len({decoy.node_id, first.node_id, second.node_id}) == 3
    assert fn_params(session, decoy, first, second) == ["q#1", "q", "q#2"]


def test_a_declaration_stays_the_callers_identity_after_a_derivation_of_that_name() -> None:
    """Reverse order: `name="q"` is a declaration, not a request — it interns with the `q` there."""
    session, x = toy_session()

    derived = x.map(q)
    declared = x.map(zero, name="q")

    assert derived.node_id == declared.node_id
    assert fn_params(session, derived) == ["q"]


def test_two_callables_under_one_declared_name_are_still_one_node() -> None:
    """O4's control, unchanged by the name space: equal declared names intern."""
    session, x = toy_session()

    shared = x.map(zero, name="shared")
    also_shared = x.map(scale, name="shared")

    assert shared.node_id == also_shared.node_id
    assert session.node_count() == 2
