"""M10 — fused stages are EXECUTABLE through `GraphStore.nodes()` (finding A.2).

Before this milestone, `nodes()` exposed a stage only as `n_members`, so no executor could
evaluate the reduced IR — execution had to re-walk the un-reduced op log. This suite pins the
introspection contract that IR-driven execution (graphed M10) builds on: every stage member is a
decoded `(kind, name, params, inputs)` record, and a generic interpreter over the REDUCED node
list reproduces the un-reduced semantics with one dispatch per fused op.
"""

from __future__ import annotations

from typing import Any

from graphed_core import GraphStore


def _toy_eval(store: GraphStore, seeds: dict[str, int]) -> list[int]:
    """A tiny integer interpreter over `nodes()` — the shape every IR evaluator takes."""

    def apply(name: str, vals: list[int], params: dict[str, Any]) -> int:
        if name == "add":
            return sum(vals)
        if name == "mul":
            out = 1
            for v in vals:
                out *= v
            return out
        if name == "neg":
            return -vals[0]
        if name == "inc":
            return vals[0] + 1
        return vals[0]

    vals: list[int] = []
    for nd in store.nodes():
        if nd["kind"] == "source":
            vals.append(seeds[nd["name"]])
        elif nd["kind"] in ("op", "reduction"):
            vals.append(apply(nd["name"], [vals[i] for i in nd["inputs"]], nd["params"]))
        elif nd["kind"] == "stage":
            ins = [vals[i] for i in nd["inputs"]]
            mvals: list[int] = []
            for m in nd["members"]:
                mins = [ins[i] if tag == "input" else mvals[i] for tag, i in m["inputs"]]
                mvals.append(apply(m["name"], mins, m["params"]))
            vals.append(mvals[-1])
        else:  # external — identity in the toy semantics
            vals.append(vals[nd["inputs"][0]])
    return [vals[i] for i in store.outputs()]


def _diamond() -> tuple[GraphStore, int]:
    s = GraphStore()
    x = s.add_source("x")
    a = s.add_op("inc", [x])
    left = s.add_op("inc", [a])
    right = s.add_op("neg", [a])
    out = s.add_op("add", [left, right])
    return s, out


def test_stage_members_are_decoded_and_executable() -> None:
    s = GraphStore()
    x = s.add_source("events")
    a = s.add_op("inc", [x], {"step": 1})
    b = s.add_op("inc", [a], {"step": 2})
    reduced, report = s.reduce(outputs=[b])
    assert report["stages"] == 1
    stage = next(n for n in reduced.nodes() if n["kind"] == "stage")
    assert len(stage["members"]) == stage["n_members"] == 2
    m0, m1 = stage["members"]
    # decoded executable form: name + typed params + resolved refs
    assert m0["name"] == "inc" and m0["params"] == {"step": 1}
    assert m1["name"] == "inc" and m1["params"] == {"step": 2}
    assert m0["inputs"] == [("input", 0)]
    assert m1["inputs"] == [("member", 0)]


def test_reduced_ir_evaluates_to_unreduced_semantics() -> None:
    s, out = _diamond()
    # the unreduced reference: serialize FOR the output (the artifact carries the flag the
    # toy interpreter reads), then evaluate
    expect = _toy_eval(GraphStore.deserialize(s.serialize(outputs=[out])), {"x": 5})
    assert expect == [1]  # ((x+1)+1) + (-(x+1)) = 1
    for maximal in (False, True):
        reduced, _ = s.reduce(maximal_fusion=maximal, outputs=[out])
        assert _toy_eval(reduced, {"x": 5}) == expect, f"maximal_fusion={maximal}"


def test_member_param_types_round_trip() -> None:
    # int / float / bool / str params survive the stage-member decode with their types
    s = GraphStore()
    x = s.add_source("x")
    a = s.add_op("cut", [x], {"thr": 30.5, "keep": True, "field": "Muon.pt", "n": 2})
    reduced, _ = s.reduce(outputs=[a])
    member = next(n for n in reduced.nodes() if n["kind"] == "stage")["members"][0]
    assert member["params"] == {"thr": 30.5, "keep": True, "field": "Muon.pt", "n": 2}
    assert type(member["params"]["n"]) is int
    assert type(member["params"]["thr"]) is float
    assert type(member["params"]["keep"]) is bool


def test_outputs_are_exposed_in_request_order() -> None:
    # [freeze-M22-1 respin: with the mark_output mutator removed, outputs() reflects what a
    # reduced/deserialized artifact carries — in the order the compile request gave them]
    s = GraphStore()
    x = s.add_source("x")
    a = s.add_op("inc", [x])
    b = s.add_op("neg", [x])
    assert s.outputs() == []  # an unreduced store has no outputs; there is no setter
    reduced, _ = s.reduce(outputs=[b, a])
    assert len(reduced.outputs()) == 2  # both carried, in request order


def test_stage_members_survive_the_durable_codec() -> None:
    # serialize -> deserialize preserves the decoded member view (IR-driven workers deserialize)
    s2, out2 = _diamond()
    reduced, _ = s2.reduce(outputs=[out2])
    back = GraphStore.deserialize(bytes(reduced.serialize()))
    assert back.nodes() == reduced.nodes()
    assert _toy_eval(back, {"x": 9}) == _toy_eval(reduced, {"x": 9})
