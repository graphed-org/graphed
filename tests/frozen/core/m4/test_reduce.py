"""M4 optimizer: reduction properties through the PyO3 boundary (plan M4).

(Semantic equivalence of reduced vs un-reduced under a toy interpreter is proven in the Rust suite;
the full numpy/awkward-backend executor equivalence is M7.)
"""

from __future__ import annotations

import graphed.core as gc


def _chain(n: int) -> gc.GraphStore:
    s = gc.GraphStore()
    cur = s.add_source("x")
    for _ in range(n):
        cur = s.add_op("inc", [cur])
    return s, cur


def test_chain_reduces_to_constant_stages_regardless_of_length() -> None:
    for n in (10, 200, 5000):
        s, out = _chain(n)
        reduced, report = s.reduce(outputs=[out])
        assert reduced.node_count() == 2  # source + one fused stage, independent of n
        assert report["stages"] == 1
        assert report["stages"] < 10


def test_canonical_analysis_reduces_to_few_stages() -> None:
    # a realistic small analysis: source -> selection ops -> reduction -> post ops -> output,
    # with many intermediate variables but only a couple of boundaries.
    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "f.root"})
    x = src
    for i in range(40):  # lots of intermediate "variables"
        x = s.add_op("select", [x], {"step": i})
    red = s.add_reduction("sum", [x])
    y = red
    for i in range(20):
        y = s.add_op("post", [y], {"step": i})
    reduced, report = s.reduce(outputs=[y])
    assert report["stages"] < 10  # regardless of intermediate-variable count
    # source + pre-stage + reduction + post-stage = 4
    assert reduced.node_count() == 4


def test_commuted_adds_merge_via_equality_saturation() -> None:
    s = gc.GraphStore()
    a = s.add_source("a")
    b = s.add_source("b")
    ab = s.add_op("add", [a, b])
    ba = s.add_op("add", [b, a])
    assert ab != ba  # distinct before reduction
    reduced, report = s.reduce(outputs=[ab, ba])
    assert report["stages"] == 1  # egg commutativity + CSE merge them
    assert reduced.node_count() == 3  # a, b, one stage


def test_additive_identity_is_simplified_away() -> None:
    s = gc.GraphStore()
    x = s.add_source("x")
    out = s.add_op("add", [x], {"scalar": 0.0, "side": "r"})
    _reduced, report = s.reduce(outputs=[out])
    assert report["stages"] == 0  # x + 0 -> x


def test_fusion_never_crosses_a_boundary() -> None:
    s = gc.GraphStore()
    src = s.add_source("x")
    pre = s.add_op("inc", [src])
    red = s.add_reduction("sum", [pre])
    post = s.add_op("inc", [red])
    _reduced, report = s.reduce(outputs=[post])
    assert report["stages"] == 2  # the reduction boundary splits the two stages


def test_dead_code_is_eliminated() -> None:
    s = gc.GraphStore()
    a = s.add_source("a")
    live = s.add_op("inc", [a])
    s.add_op("neg", [a])  # dead: never an output
    reduced, report = s.reduce(outputs=[live])
    assert report["reachable_nodes"] == 2  # a + live (dead op dropped)
    assert reduced.node_count() == 2


def test_reduction_is_byte_deterministic() -> None:
    def build() -> str:
        s = gc.GraphStore()
        a = s.add_source("a")
        b = s.add_source("b")
        c = s.add_op("add", [a, b])
        d = s.add_op("inc", [c])
        return s.reduce(outputs=[d])[0].to_dot()

    assert build() == build()


def test_reduction_report_has_expected_keys() -> None:
    s, out = _chain(5)
    report = s.reduction_report(outputs=[out])
    for key in (
        "input_nodes",
        "reachable_nodes",
        "canonical_nodes",
        "stages",
        "reduced_nodes",
        "boundary_nodes",
    ):
        assert key in report
    assert report["input_nodes"] == 6  # source + 5 ops


def test_incremental_reduction_matches_full() -> None:
    s, out = _chain(50)
    full = s.reduce(outputs=[out])[1]
    incr = s.reduce_incremental(outputs=[out])[1]
    assert full == incr
