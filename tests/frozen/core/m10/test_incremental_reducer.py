"""M10 — `IncrementalReducer` is genuinely incremental (finding A.1).

The M4 `reduce_incremental()` alias satisfied an equality test but did no incremental work. This
suite pins the properties an alias CANNOT satisfy: per-step work equals the delta (never the
history), the cumulative work over any number of steps equals the node count, and the maintained
canonical form stays concise while building. Equivalence with one-shot `reduce` is pinned
byte-for-byte through the M8 codec.
"""

from __future__ import annotations

from graphed_core import GraphStore, IncrementalReducer


def _chain_with_twins(n: int) -> GraphStore:
    """A realistic build: a chain plus commuted duplicate work and an identity op."""
    s = GraphStore()
    a = s.add_source("a")
    b = s.add_source("b")
    ab = s.add_op("add", [a, b])
    ba = s.add_op("add", [b, a])  # commuted twin of ab
    one = s.add_op("mul", [ab], {"scalar": 1.0, "side": "r"})  # x * 1 -> x
    cur = s.add_op("mul", [one, ba])
    for i in range(n):
        cur = s.add_op("inc", [cur], {"i": i})
    s.mark_output(cur)
    return s


def test_per_step_work_is_the_delta_never_the_history() -> None:
    s = GraphStore()
    src = s.add_source("x")
    r = IncrementalReducer()
    assert r.step(s) == 1  # the source
    cur = src
    for i in range(200):
        cur = s.add_op("inc", [cur], {"i": i})
        did = r.step(s)
        assert did == 1, f"step {i}: processed {did} nodes, expected only the delta"
    # cumulative work == node count: each node touched exactly once across the whole build
    assert r.total_work() == s.node_count() == 201
    assert r.watermark() == s.node_count()


def test_canonical_form_stays_concise_while_building() -> None:
    # duplicate + commuted + identity work collapses AT RECORD TIME, not at the end
    s = GraphStore()
    a = s.add_source("a")
    b = s.add_source("b")
    r = IncrementalReducer()
    for _ in range(50):
        ab = s.add_op("add", [a, b])
        ba = s.add_op("add", [b, a])  # commuted -> same canonical node
        s.add_op("mul", [ab], {"scalar": 1.0, "side": "r"})  # identity -> no canonical node
        s.add_op("mul", [ba], {"scalar": 1.0, "side": "l"})
        r.step(s)
    # interning already dedups exact repeats; the reducer additionally collapses the commuted
    # orientation and the identity wrappers: only a, b, add remain canonical
    assert r.canonical_count() == 3


def test_finalize_is_byte_identical_to_one_shot_reduce() -> None:
    built = _chain_with_twins(25)
    r = IncrementalReducer()
    r.step(built)
    inc_store, _ = r.finalize(built)
    full_store, _ = built.reduce()
    assert inc_store.serialize() == full_store.serialize()


def test_finalize_after_many_small_steps_matches_one_big_step() -> None:
    a = _chain_with_twins(40)
    ra = IncrementalReducer()
    ra.step(a)  # one big step

    b = GraphStore()
    rb = IncrementalReducer()
    x = b.add_source("a")
    rb.step(b)
    y = b.add_source("b")
    rb.step(b)
    ab = b.add_op("add", [x, y])
    rb.step(b)
    ba = b.add_op("add", [y, x])
    one = b.add_op("mul", [ab], {"scalar": 1.0, "side": "r"})
    cur = b.add_op("mul", [one, ba])
    rb.step(b)
    for i in range(40):
        cur = b.add_op("inc", [cur], {"i": i})
        rb.step(b)
    b.mark_output(cur)

    assert ra.finalize(a)[0].serialize() == rb.finalize(b)[0].serialize()


def test_incremental_reduction_is_deterministic() -> None:
    def run() -> bytes:
        s = _chain_with_twins(10)
        r = IncrementalReducer()
        r.step(s)
        return bytes(r.finalize(s)[0].serialize())

    assert run() == run()


def test_finalize_report_matches_reduce_shape() -> None:
    s = _chain_with_twins(5)
    r = IncrementalReducer()
    _, report = r.finalize(s)  # finalize() consumes any unstepped delta itself
    for key in ("stages", "reduced_nodes", "boundary_nodes"):
        assert key in report
    assert report["stages"] >= 1
