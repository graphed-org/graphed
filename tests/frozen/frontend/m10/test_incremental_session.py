"""M10 — `Session(incremental=True)` maintains the reduced view while building (finding A.1).

The session steps a core `IncrementalReducer` on every record, so the canonical form exists at all
times and compile finishes with one linear pass — no whole-history optimization at the end. Pins:
byte-identical output vs the one-shot path, the incrementality witness (total reducer work equals
the node count regardless of build shape), and that the default session is untouched.
"""

from __future__ import annotations

from m10_toy import CountingListBackend, from_list

from graphed import Array, Session, compile_ir


def _build(session: Session, n: int = 30) -> Array:
    a = from_list(session, "a", [1.0, 2.0])
    b = from_list(session, "b", [3.0, 4.0])
    cur = a + b
    for _ in range(n):
        cur = (cur * 1.0) + (b + a)  # identity + commuted-twin chaff, folded as it is recorded
    return cur.reduce("sum")


def test_incremental_serialized_ir_is_byte_identical_to_one_shot() -> None:
    s_inc = Session(CountingListBackend(), incremental=True)
    out_inc = _build(s_inc)
    s_one = Session(CountingListBackend())
    out_one = _build(s_one)
    assert s_inc.serialized_ir(out_inc) == s_one.serialized_ir(out_one)
    # and through the compile path
    s_inc2 = Session(CountingListBackend(), incremental=True)
    s_one2 = Session(CountingListBackend())
    assert compile_ir(s_inc2, _build(s_inc2)).ir == compile_ir(s_one2, _build(s_one2)).ir


def test_reduction_state_witnesses_incrementality() -> None:
    s = Session(CountingListBackend(), incremental=True)
    out = _build(s)
    state = s.reduction_state()
    assert state is not None
    # every interned node consumed exactly once, no matter how many records stepped the reducer
    assert state["watermark"] == state["total_work"] == s.node_count()
    # the maintained canonical form has folded the identity/commuted chaff away
    assert state["canonical_count"] < s.node_count()
    # the state is live: more records advance the watermark by exactly the delta
    before = state["watermark"]
    from_list(s, "c", [9.0])
    after = s.reduction_state()
    assert after is not None and after["watermark"] == before + 1
    del out


def test_default_session_is_unchanged() -> None:
    s = Session(CountingListBackend())
    assert s.reduction_state() is None
    out = _build(s)
    # the non-incremental compile path still reduces one-shot and stays deterministic
    assert s.serialized_ir(out) == s.serialized_ir(out)


def test_incremental_session_materializes_identically() -> None:
    s_inc = Session(CountingListBackend(), incremental=True)
    s_one = Session(CountingListBackend())
    assert s_inc.materialize(_build(s_inc)) == s_one.materialize(_build(s_one))
