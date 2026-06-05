"""Complex topologies through the frontend (plans M2/M3): a diamond/star/nested graph must intern a
shared sub-expression to ONE node, and `materialize` must evaluate that shared node EXACTLY once
(never re-executing a shared sub-graph per consumer — a dask failure mode), with correct results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from graphed_core import PayloadDescriptor

from graphed import Array, Session


class _Counting:
    """A tiny integer-arithmetic backend that counts every op evaluation."""

    def __init__(self) -> None:
        self.op_evals = 0

    def op_form(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> str:
        return "num"

    def eval_stage(self, op: str, inputs: Sequence[object], params: Mapping[str, object]) -> object:
        self.op_evals += 1
        a = inputs[0]
        if "scalar" in params:  # array OP scalar
            b = params["scalar"]
            a, b = (b, a) if params.get("side") == "l" else (a, b)
        else:
            b = inputs[1] if len(inputs) > 1 else 0
        if op == "add":
            return a + b  # type: ignore[operator]
        if op == "sub":
            return a - b  # type: ignore[operator]
        if op == "mul":
            return a * b  # type: ignore[operator]
        return a

    def boundary_ops(self) -> frozenset[str]:
        return frozenset({"source"})

    def project(self, op: str, used: object, params: Mapping[str, object]) -> object:
        return used

    def external_payload(self, op: str, params: Mapping[str, object]) -> PayloadDescriptor | None:
        return None


def _src(s: Session, name: str, value: int) -> Array:
    return s.source(name, form="num", data=value)


def test_diamond_interns_apex_once_and_evaluates_it_once() -> None:
    s = Session(_Counting())
    x = _src(s, "x", 4)
    y = _src(s, "y", 3)
    apex = x + y  # 7 — reused by both branches
    out = (apex * x) + (apex + y)  # (7*4) + (7+3) = 28 + 10 = 38
    # apex interned once: x, y, apex, left(mul), right(add), out(add) = 6 nodes
    assert s.node_count() == 6
    assert s.materialize(out) == 38
    # 4 op nodes (apex, left, right, out), each evaluated once — the apex is NOT recomputed per branch
    assert s.backend.op_evals == 4


def test_star_hub_evaluated_once_for_many_consumers() -> None:
    s = Session(_Counting())
    n = 20
    x = _src(s, "x", 2)
    y = _src(s, "y", 5)
    hub = x + y  # 7 — feeds n consumers
    out = hub
    for _ in range(n):
        out = out + hub  # each consumer reuses the hub
    # hub computed once; despite n+1 references it is a single node evaluated a single time
    assert s.materialize(out) == 7 * (n + 1)
    assert s.backend.op_evals == s.node_count() - 2  # every op node once (2 sources excluded)


def test_repeated_subexpression_adds_no_new_nodes() -> None:
    s = Session(_Counting())
    x = _src(s, "x", 1)
    y = _src(s, "y", 2)
    first = (x + y) * x
    before = s.node_count()
    second = (x + y) * x  # identical structure -> interns to the SAME nodes
    assert s.node_count() == before
    assert first.node_id == second.node_id


def test_nested_diamonds_materialize_correctly_each_node_once() -> None:
    s = Session(_Counting())
    x = _src(s, "x", 10)
    one = _src(s, "one", 1)
    v = x
    for _ in range(5):
        v = (v + one) - (v - one)  # = 2 each level, independent of v; v fans out to two branches
    assert s.materialize(v) == 2
    # no shared node is recomputed: total op evals == number of op nodes
    assert s.backend.op_evals == s.node_count() - 2  # exclude the two sources
