"""M10 — the sound rule set covers every symmetric op in the vocabulary (finding C.7), and the
optimizer token encoding is injective (latent collision bug).

Before this milestone only `add`/`mul` had commute rules, so `a & b` vs `b & a` (etc.) never
merged. And `ParamMap::token()` was NOT injective: `{a: "x;b=i1"}` and `{a: "x", b: 1}` encoded to
the same token, which could make the optimizer rebuild a boundary node with the wrong params.
"""

from __future__ import annotations

from graphed_core import GraphStore

# every symmetric binary op the frontend records (mirrors the shared Rust SYMMETRIC_OPS constant)
SYMMETRIC_OPS = ["add", "mul", "and", "or", "eq", "ne", "maximum", "minimum"]


def test_commuted_twins_merge_for_every_symmetric_op() -> None:
    for op in SYMMETRIC_OPS:
        s = GraphStore()
        a = s.add_source("a")
        b = s.add_source("b")
        ab = s.add_op(op, [a, b])
        ba = s.add_op(op, [b, a])
        assert ab != ba  # distinct when recorded
        s.mark_output(ab)
        s.mark_output(ba)
        _, report = s.reduce()
        assert report["stages"] == 1, f"{op}: commuted twins must merge into one stage"


def test_asymmetric_ops_do_not_merge() -> None:
    for op in ("sub", "div", "lt", "ge", "filter"):
        s = GraphStore()
        a = s.add_source("a")
        b = s.add_source("b")
        s.mark_output(s.add_op(op, [a, b]))
        s.mark_output(s.add_op(op, [b, a]))
        _, report = s.reduce()
        assert report["stages"] == 2, f"{op} is not symmetric; commuting it would be unsound"


def test_param_tokens_are_injective_for_hostile_strings() -> None:
    # {a: "x;b=i1"} vs {a: "x", b: 1} collided under the old encoding; they must stay distinct
    # through reduction (same single-input op, same downstream consumer — only params differ).
    s = GraphStore()
    x = s.add_source("x")
    hostile = s.add_op("tag", [x], {"a": "x;b=i1"})
    benign = s.add_op("tag", [x], {"a": "x", "b": 1})
    assert hostile != benign
    s.mark_output(hostile)
    s.mark_output(benign)
    reduced, _ = s.reduce()
    # both ops survive reduction with their own params intact
    blob = bytes(reduced.serialize())
    back = GraphStore.deserialize(blob)
    stages = [n for n in back.nodes() if n["kind"] == "stage"]
    params = sorted(str(m["params"]) for st in stages for m in st["members"])
    assert len(params) == 2 and params[0] != params[1]


def test_separator_characters_in_params_round_trip_through_members() -> None:
    s = GraphStore()
    x = s.add_source("x")
    weird: dict[str, int | float | bool | str] = {"expr": "pt>30; |eta|<2.4", "label": "a=b%c"}
    s.mark_output(s.add_op("cut", [x], weird))
    reduced, _ = s.reduce()
    member = next(n for n in reduced.nodes() if n["kind"] == "stage")["members"][0]
    assert member["params"] == weird
