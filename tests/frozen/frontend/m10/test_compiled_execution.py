"""M10 — IR-driven execution: the REDUCED serialized IR is what executes (finding A.2).

Pins the properties the old per-partition re-recording path could not have: evaluation works from
the compiled bytes alone (no Session, no analysis function), costs one backend dispatch per
REDUCED op (not per recorded op), retargets to alternate inputs without recompiling, and resolves
opaque External payloads explicitly by content hash (failing loudly when one is missing).
"""

from __future__ import annotations

import pickle

import graphed_core
import pytest
from m10_toy import CountingListBackend, from_list

from graphed import Array, CompiledGraph, GraphedError, Session, compile_ir, evaluate_ir


def _record(session: Session) -> Array:
    """a + b twice over (commuted), through an identity, summed — reduction collapses all of it."""
    a = from_list(session, "a", [1.0, 2.0, 3.0])
    b = from_list(session, "b", [10.0, 20.0, 30.0])
    ab = a + b
    ba = b + a  # commuted twin -> same canonical node after reduction
    kept = ab * 1.0  # identity -> eliminated by reduction
    return (kept + ba).reduce("sum")


def test_compiled_evaluation_matches_materialize() -> None:
    s = Session(CountingListBackend())
    out = _record(s)
    expect = s.materialize(out)  # the M2 reference evaluator (un-reduced walk)
    compiled = compile_ir(s, out)
    got = evaluate_ir(compiled, CountingListBackend(), {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    assert got == [expect] == [132.0]


def test_evaluation_needs_only_bytes_no_session_no_user_code() -> None:
    s = Session(CountingListBackend())
    out = _record(s)
    blob = pickle.dumps(compile_ir(s, out))
    del s, out  # nothing of the recording survives

    compiled = pickle.loads(blob)
    assert isinstance(compiled, CompiledGraph)
    assert set(compiled.source_names) == {"a", "b"}
    got = compiled.evaluate(CountingListBackend(), {"a": [1.0], "b": [2.0]})
    assert got == [6.0]  # (1+2)+(2+1) summed


def test_dispatch_count_is_the_reduced_graph_not_the_recorded_one() -> None:
    s = Session(CountingListBackend())
    out = _record(s)
    # recorded: add, add(commuted), mul(identity), add, sum -> 5 ops
    walker = CountingListBackend()
    s2 = Session(walker)
    out2 = _record(s2)
    s2.materialize(out2)
    assert len(walker.calls) == 5

    runner = CountingListBackend()
    evaluate_ir(compile_ir(s, out), runner, {"a": [1.0], "b": [2.0]})
    # reduced: ONE canonical add + final add + sum -> 3 dispatches (commuted twin merged via the
    # sound commute rule, identity eliminated)
    assert len(runner.calls) == 3
    assert runner.calls.count("sum") == 1


def test_compile_once_run_on_alternate_inputs() -> None:
    s = Session(CountingListBackend())
    out = _record(s)
    compiled = compile_ir(s, out)
    r1 = evaluate_ir(compiled, CountingListBackend(), {"a": [1.0], "b": [2.0]})
    r2 = evaluate_ir(compiled, CountingListBackend(), {"a": [5.0, 6.0], "b": [7.0, 8.0]})
    assert r1 == [6.0] and r2 == [52.0]
    # a source may be bound to a zero-arg loader (the lazy-read shape host readers use)
    r3 = evaluate_ir(compiled, CountingListBackend(), {"a": lambda: [1.0], "b": lambda: [2.0]})
    assert r3 == r1


def test_missing_source_fails_loudly() -> None:
    s = Session(CountingListBackend())
    out = _record(s)
    compiled = compile_ir(s, out)
    with pytest.raises(GraphedError, match="no data bound for source 'b'"):
        evaluate_ir(compiled, CountingListBackend(), {"a": [1.0]})


def test_external_payloads_resolve_by_content_hash_and_fail_loudly() -> None:
    backend = CountingListBackend()
    s = Session(backend)
    a = from_list(s, "a", [1.0, 2.0])

    def times10(xs: object) -> object:
        assert isinstance(xs, list)
        return [x * 10 for x in xs]

    out = a.map(times10).reduce("sum")
    compiled = compile_ir(s, out)

    with pytest.raises(GraphedError, match="needs an evaluator"):
        evaluate_ir(compiled, CountingListBackend(), {"a": [1.0, 2.0]})

    # the payload's content hash is IN the compiled IR — resolve the evaluator from there, the way
    # a real deployment binds payload-backed evaluators (no session needed)
    store = graphed_core.GraphStore.deserialize(compiled.ir)
    (chash,) = [n["descriptor"]["content_hash"] for n in store.nodes() if n["kind"] == "external"]
    got = evaluate_ir(
        compiled,
        CountingListBackend(),
        {"a": [1.0, 2.0]},
        externals={chash: times10},
    )
    assert got == [30.0]


def test_compiled_graph_is_deterministic() -> None:
    def build() -> bytes:
        s = Session(CountingListBackend())
        return compile_ir(s, _record(s)).ir

    assert build() == build()


def test_maximal_fusion_compile_evaluates_identically() -> None:
    s = Session(CountingListBackend())
    out = _record(s)
    plain = compile_ir(s, out)
    s2 = Session(CountingListBackend())
    out2 = _record(s2)
    maximal = compile_ir(s2, out2, maximal_fusion=True)
    sources = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    assert evaluate_ir(plain, CountingListBackend(), sources) == evaluate_ir(
        maximal, CountingListBackend(), sources
    )
