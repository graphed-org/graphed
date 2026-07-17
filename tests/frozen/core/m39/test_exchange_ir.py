"""M39 — the ``Exchange`` boundary NodeKey variant (plan §2.1, §2.4).

An ``Exchange`` is a pure data-movement boundary with exactly one logical input; its identity is its
``scheme`` ParamMap (scheme=hash|range|coalesce|split, key, parts, seed, …). It is NOT an ``Op``, so
it interns/CSEs like every other node, **ends a stage** (fusion never crosses it), survives reduce as
itself, and serializes byte-identically. These tests pin that behavior at the IR layer; the routing
*rule* it carries is pinned in the backend golden-vector suite (graphed-awkward/graphed-numpy m39).

Pinned Python surface (test-author decision, faithful to the §2.1 enum which has no ``name``):
``GraphStore.add_exchange(inputs: list[int], params) -> int`` and ``nodes()[i]["kind"] == "exchange"``.
"""

from __future__ import annotations

from graphed.core import GraphStore


def _chain_with_exchange(parts: int = 8) -> tuple[GraphStore, int, int]:
    """src -> pt -> cut -> EXCHANGE(hash) -> shift -> scale -> sum(output)."""
    g = GraphStore()
    src = g.add_source("events", {"uri": "f.root"})
    pt = g.add_op("pt", [src])
    cut = g.add_op("gt", [pt], {"thr": 30.0})
    xchg = g.add_exchange([cut], {"scheme": "hash", "key": "__joinkey__", "parts": parts})
    shift = g.add_op("add1", [xchg])
    scale = g.add_op("mul2", [shift])
    out = g.add_reduction("sum", [scale])
    return g, xchg, out


def _kinds(g: GraphStore) -> list[str]:
    return [n["kind"] for n in g.nodes()]


def test_add_exchange_records_a_boundary_node_with_its_scheme() -> None:
    g, xchg, _ = _chain_with_exchange(parts=8)
    node = g.nodes()[xchg]
    assert node["kind"] == "exchange", "Exchange must be its own node kind, not an op/reduction"
    assert node["inputs"] == [xchg - 1], "Exchange has exactly its one logical input"
    # the scheme params ARE the node's identity and must round-trip through nodes()
    assert node["params"]["scheme"] == "hash"
    assert node["params"]["key"] == "__joinkey__"
    assert node["params"]["parts"] == 8


def test_exchange_interns_and_cses_like_any_node() -> None:
    # two structurally identical exchanges over the SAME input are ONE node (hash-cons / CSE, M1).
    g = GraphStore()
    src = g.add_source("events", {"uri": "f.root"})
    a = g.add_exchange([src], {"scheme": "hash", "key": "k", "parts": 4})
    b = g.add_exchange([src], {"scheme": "hash", "key": "k", "parts": 4})
    assert a == b, "identical Exchange nodes must intern to one id (CSE falls out of hash-consing)"
    assert g.node_count() == 2  # source + one exchange


def test_exchange_scheme_participates_in_structural_identity() -> None:
    # different scheme params -> different node (params must be in the structural hash, not ignored).
    g = GraphStore()
    src = g.add_source("events", {"uri": "f.root"})
    p8 = g.add_exchange([src], {"scheme": "hash", "key": "k", "parts": 8})
    p16 = g.add_exchange([src], {"scheme": "hash", "key": "k", "parts": 16})
    coalesce = g.add_exchange([src], {"scheme": "coalesce", "target_bytes": 1024})
    assert len({p8, p16, coalesce}) == 3, "parts/scheme differences must be distinct nodes"


def test_exchange_ends_a_stage_and_survives_reduce() -> None:
    # THE boundary witness: the two op-runs on each side fuse into SEPARATE stages, the Exchange is
    # emitted as itself between them, and it is NOT absorbed into an op run. An implementation that
    # made Exchange a fusible Op would fuse straight through it (one stage, no exchange node).
    g, _, out = _chain_with_exchange(parts=8)
    reduced, _report = g.reduce(outputs=[out])
    kinds = _kinds(reduced)
    assert kinds.count("exchange") == 1, "the Exchange boundary must survive reduction as itself"
    assert kinds.count("stage") == 2, "the boundary must SPLIT the op runs into two fused stages"
    # a boundary neither fuses across nor vanishes: source + 2 stages + exchange + reduction
    assert reduced.node_count() == 5


def test_reduction_over_an_exchange_is_byte_deterministic() -> None:
    # the M8/M4 determinism gate extends to the new variant: identical builds -> identical reduced dot.
    a, _, oa = _chain_with_exchange(parts=8)
    b, _, ob = _chain_with_exchange(parts=8)
    assert a.reduce(outputs=[oa])[0].to_dot() == b.reduce(outputs=[ob])[0].to_dot()
