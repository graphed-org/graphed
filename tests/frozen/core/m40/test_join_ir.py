"""M40 — the ``Join`` boundary NodeKey variant (plan §2.1, contract E2).

A ``Join`` is a relational data-movement boundary with exactly **two** logical inputs (build side,
probe side); its identity is its ``scheme`` ParamMap (``how``, ``on``, …) together with the ORDER of
those two inputs. Like every non-``Op`` node it interns/CSEs (M1 hash-consing), **ends a stage**
(fusion never crosses it), and survives reduction as itself — a Join is a boundary op (plan A.6), so
`is_boundary` needs no change (it is ``!Op``). These tests pin that at the IR layer; the relational
*semantics* it carries are pinned in the frontend/backend M40 suites.

Pinned Python surface (mirrors M39 ``add_exchange``): ``GraphStore.add_join(inputs, params) -> int``
with ``len(inputs) == 2``; ``nodes()[i]["kind"] == "join"``; ``params`` is the scheme map.
"""

from __future__ import annotations

from graphed.core import GraphStore

_SCHEME = {"how": "inner", "on": "event"}


def _two_sided_join() -> tuple[GraphStore, int, int]:
    """left: src_l->a->b  ; right: src_r->c->d ; JOIN(b, d) -> e->f -> sum(output).

    Two 2-op runs feed the Join; a 2-op run follows it. Under single-use fusion each run collapses
    to ONE stage, so the reduced graph is: 2 sources + 3 stages + 1 join + 1 reduction = 7 nodes.
    """
    g = GraphStore()
    lsrc = g.add_source("left", {"uri": "l.root"})
    rsrc = g.add_source("right", {"uri": "r.root"})
    a = g.add_op("add1", [lsrc])
    b = g.add_op("mul2", [a])
    c = g.add_op("add1", [rsrc])
    d = g.add_op("mul2", [c])
    j = g.add_join([b, d], _SCHEME)
    e = g.add_op("add1", [j])
    f = g.add_op("mul2", [e])
    out = g.add_reduction("sum", [f])
    return g, j, out


def _kinds(g: GraphStore) -> list[str]:
    return [n["kind"] for n in g.nodes()]


def test_add_join_records_a_two_input_boundary_node_with_its_scheme() -> None:
    g, j, _ = _two_sided_join()
    node = g.nodes()[j]
    assert node["kind"] == "join", "Join must be its own node kind, not an op/reduction/exchange"
    # a Join has EXACTLY two logical inputs — build side then probe side (order is identity).
    assert len(node["inputs"]) == 2, "a Join is a two-input boundary (unlike Exchange's one)"
    assert node["inputs"] == [j - 3, j - 1], "the two op-run heads (b, d) are the join's inputs"
    # the scheme params ARE the node's identity and must round-trip through nodes()
    assert node["params"]["how"] == "inner"
    assert node["params"]["on"] == "event"


def test_join_interns_and_cses_like_any_node() -> None:
    # two structurally identical joins over the SAME two inputs are ONE node (hash-cons / CSE, M1).
    g = GraphStore()
    lsrc = g.add_source("left")
    rsrc = g.add_source("right")
    a = g.add_join([lsrc, rsrc], _SCHEME)
    b = g.add_join([lsrc, rsrc], _SCHEME)
    assert a == b, "identical Join nodes must intern to one id (CSE falls out of hash-consing)"
    assert g.node_count() == 3, "two sources + one interned join"


def test_join_scheme_and_input_order_participate_in_structural_identity() -> None:
    # different scheme -> different node; and SWAPPING build/probe -> different node (a Join is
    # asymmetric: left-outer of (L,R) is not the same computation as (R,L)). A wrong impl that
    # sorts/canonicalizes the two inputs would wrongly merge lr and rl.
    g = GraphStore()
    lsrc = g.add_source("left")
    rsrc = g.add_source("right")
    lr = g.add_join([lsrc, rsrc], _SCHEME)
    rl = g.add_join([rsrc, lsrc], _SCHEME)
    left_outer = g.add_join([lsrc, rsrc], {"how": "left", "on": "event"})
    assert len({lr, rl, left_outer}) == 3, "scheme AND input order must both distinguish joins"


def test_join_ends_stages_and_survives_reduce() -> None:
    # THE boundary witness: the three op-runs fuse into THREE separate stages, the Join is emitted as
    # itself between them, and it is NOT absorbed into an op run. If Join were a fusible Op, fusion
    # would run straight through it: the whole graph collapses to ONE stage with NO join node.
    g, _, out = _two_sided_join()
    reduced, _report = g.reduce(outputs=[out])
    kinds = _kinds(reduced)
    assert kinds.count("join") == 1, "the Join boundary must survive reduction as itself"
    assert kinds.count("stage") == 3, "the boundary must split the op runs into three fused stages"
    # 2 sources + 3 stages + join + reduction; the boundary neither fuses across nor vanishes.
    assert reduced.node_count() == 7
    # the surviving join still has its two inputs, each pointing at a fused stage (not an op).
    jnode = next(n for n in reduced.nodes() if n["kind"] == "join")
    assert len(jnode["inputs"]) == 2
    assert all(reduced.nodes()[i]["kind"] == "stage" for i in jnode["inputs"])


def test_dce_keeps_a_reachable_join_and_drops_an_unreachable_one() -> None:
    # DCE = reachability from outputs (plan M4). A join on the path to the marked output survives;
    # a second join off to the side, never reached, is dropped. A wrong DCE keeps the dead join.
    g = GraphStore()
    lsrc = g.add_source("left")
    rsrc = g.add_source("right")
    live = g.add_join([lsrc, rsrc], _SCHEME)
    dead = g.add_join([rsrc, lsrc], _SCHEME)  # distinct (swapped order); never on a path to output
    out = g.add_reduction("sum", [live])
    reduced, _ = g.reduce(outputs=[out])
    assert [n["kind"] for n in reduced.nodes()].count("join") == 1, "only the reachable join survives"
    assert dead != live  # sanity: they really are two distinct nodes in the source graph


def test_reduction_over_a_join_is_byte_deterministic() -> None:
    # the M8/M4 determinism gate extends to the new variant: identical builds -> identical reduced dot.
    a, _, oa = _two_sided_join()
    b, _, ob = _two_sided_join()
    assert a.reduce(outputs=[oa])[0].to_dot() == b.reduce(outputs=[ob])[0].to_dot()
