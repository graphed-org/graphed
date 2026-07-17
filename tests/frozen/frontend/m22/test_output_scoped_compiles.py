"""M22 (graphed): compiles are output-scoped — the compile_ir accumulation footgun is FIXED.

`compile_ir(session, expr)` and `Session.serialized_ir(expr)` produce artifacts that carry
EXACTLY the requested outputs and write NO session state (freeze-M22-1: the `mark_output`
mutator is removed outright), so compiling two different
expressions from one session yields two independent single-output artifacts — each byte-identical
to what a fresh session would produce (compilation is session-history-independent). Deliberate
multi-output requests (`compile_ir(s, a, b)`) still carry both. Holds on the one-shot, the
incremental (M10), and the `optimize=False` paths.
"""

from __future__ import annotations

from m10_toy import CountingListBackend, from_list

import graphed.core
from graphed import Array, Session, compile_ir, evaluate_ir

SRC = {"x": [1.0, 2.0, 3.0]}


def _session(incremental: bool = False) -> tuple[Session, Array, Array]:
    s = Session(CountingListBackend(), incremental=incremental)
    x = from_list(s, "x", [1.0, 2.0, 3.0])
    a = (x + 1.0).reduce("sum")  # analysis A -> 9.0
    b = (x * 2.0).reduce("sum")  # analysis B -> 12.0
    return s, a, b


def _flag_count(blob: bytes) -> int:
    return sum(1 for n in graphed.core.GraphStore.deserialize(blob).nodes() if n["output"])


def test_footgun_reproducer_each_compile_carries_exactly_one_output() -> None:
    s, a, b = _session()
    ca = compile_ir(s, a)  # the second compile used to inherit the first's output mark
    cb = compile_ir(s, b)
    assert evaluate_ir(ca, CountingListBackend(), SRC) == [9.0]
    assert evaluate_ir(cb, CountingListBackend(), SRC) == [12.0]
    assert _flag_count(ca.ir) == 1 and _flag_count(cb.ir) == 1  # never the union


def test_compiles_are_session_history_independent_byte_for_byte() -> None:
    s, a, b = _session()
    ca_first = compile_ir(s, a).ir
    cb = compile_ir(s, b).ir
    ca_again = compile_ir(s, a).ir  # after compiling B: identical — no accumulation either way
    fresh_s, _fresh_a, fresh_b = _session()
    assert cb == compile_ir(fresh_s, fresh_b).ir
    assert ca_again == ca_first


def test_serialized_ir_is_output_scoped_on_optimized_and_raw_paths() -> None:
    s, a, b = _session()
    _ = s.serialized_ir(a)
    scoped = s.serialized_ir(b)
    assert _flag_count(scoped) == 1
    fresh_s, _fa, fb = _session()
    assert scoped == fresh_s.serialized_ir(fb)
    # optimize=False serializes the WHOLE store but flags only the requested output
    raw = s.serialized_ir(b, optimize=False)
    g = graphed.core.GraphStore.deserialize(raw)
    assert [n["id"] for n in g.nodes() if n["output"]] == [b.node_id]


def test_incremental_sessions_scope_outputs_identically() -> None:
    s, a, b = _session(incremental=True)
    _ = compile_ir(s, a)
    inc = compile_ir(s, b).ir
    one_shot_s, _oa, ob = _session()
    assert inc == compile_ir(one_shot_s, ob).ir  # M10 byte-identity, now per compile request
    assert evaluate_ir(inc, CountingListBackend(), SRC) == [12.0]


def test_deliberate_multi_output_compiles_still_carry_both() -> None:
    s, a, b = _session()
    both = compile_ir(s, a, b)
    assert _flag_count(both.ir) == 2
    assert evaluate_ir(both, CountingListBackend(), SRC) == [9.0, 12.0]
