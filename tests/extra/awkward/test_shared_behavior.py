"""A recorded node keeps its backend's behavior object, not a copy per node (#297)."""

from __future__ import annotations

from typing import Any, cast

import awkward as ak

import graphed
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak

BEHAVIOR: dict[Any, Any] = {f"entry{i}": i for i in range(100)}


def _tt(session: graphed.Session, x: graphed.Array) -> Any:
    return cast("AwkwardForm", session.form(x)).tt


def test_every_kept_result_holds_the_backend_behavior_object() -> None:
    session = graphed.Session(AwkwardBackend(behavior=BEHAVIOR))
    data = ak.Array([{"k": 1, "x": 1.0, "y": [1.0]}, {"k": 2, "x": 2.0, "y": []}])
    ev = from_awkward("e", data, session=session)
    other = from_awkward("o", ak.Array([{"k": 1, "z": 5.0}]), session=session)
    outs = [
        ev.map(lambda e: ak.where(e.x > 1, e.x, e.x * 3)),
        gak.join(gak.zip({"k": ev.k, "x": ev.x}), gak.zip({"k": other.k, "z": other.z}), on="k"),
    ]
    for i in range(20):
        outs += [
            gak.with_field(ev, ev.x * i, f"z{i}"),
            gak.where(ev.x > i, ev.x, ev.x * 3),
            gak.zip({"p": ev.x, "q": ev.x + i}),
        ]
    behavior = _tt(session, ev.x * 2).behavior
    assert behavior is not None and dict(behavior) == BEHAVIOR
    assert all(_tt(session, o).behavior is behavior for o in outs)


class _Inner(ak.Array):  # type: ignore[misc]
    @property
    def doubled(self) -> Any:
        return self.a * 2


INNER: dict[Any, Any] = {("*", "Inner"): _Inner}


class _Outer(ak.Array):  # type: ignore[misc]
    def inner(self, other: Any) -> Any:
        return ak.zip({"a": self.a + other}, with_name="Inner", behavior=INNER)


OUTER: dict[Any, Any] = {("*", "Outer"): _Outer}


def test_a_method_result_keeps_its_own_behavior_without_a_backend_one() -> None:
    session = graphed.Session(AwkwardBackend())
    ev = from_awkward("e", ak.Array([{"x": 1.0}, {"x": 2.0}]), session=session)
    outer = ev.map(lambda e: ak.zip({"a": e.x}, with_name="Outer", behavior=OUTER))
    inner = outer.inner(ev.x)
    assert _tt(session, inner).behavior is INNER
    assert session.form(inner.doubled).describe() == "## * float64"
