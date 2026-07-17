"""M22: output-scoped reduction & serialization (the compile_ir output-accumulation fix).

Graph outputs are a property of the COMPILE REQUEST, not store state: `reduce`, `serialize`, and
`IncrementalReducer.finalize` take an explicit `outputs=` set that is used EXACTLY, so sequential
compiles of different expressions from one store are fully independent, byte-identical to
fresh-store compiles, and safe under concurrency (compiling is a READ-ONLY operation).
[freeze-M22-1, user-authorized respin: the `mark_output` mutator is REMOVED — an unreduced store
simply has no outputs; reduced/deserialized artifacts carry the outputs their request named.]
"""

from __future__ import annotations

import threading

import pytest

import graphed.core as gc
from graphed.core import IncrementalReducer


def _two_analyses() -> tuple[gc.GraphStore, int, int]:
    """One store carrying two independent analyses, A and B, over a shared source."""
    s = gc.GraphStore()
    src = s.add_source("events", {"uri": "f.root"})
    a = s.add_op("pt", [src])
    a = s.add_reduction("sum", [a])
    b = s.add_op("eta", [src])
    b = s.add_op("abs", [b])
    return s, a, b


def _fresh_request_bytes(pick_second: bool) -> bytes:
    """Reference: a FRESH store compiled for exactly the chosen analysis."""
    s, a, b = _two_analyses()
    return bytes(s.reduce(outputs=[b if pick_second else a])[0].serialize())


def _flagged(store: gc.GraphStore) -> list[int]:
    return [n["id"] for n in store.nodes() if n["output"]]


def test_reduce_uses_exactly_the_requested_outputs() -> None:
    s, a, b = _two_analyses()
    _ = s.reduce(outputs=[a])  # an earlier compile of the OTHER analysis
    reduced, _ = s.reduce(outputs=[b])
    assert len(_flagged(reduced)) == 1  # exactly the requested output — not the union
    assert bytes(reduced.serialize()) == _fresh_request_bytes(pick_second=True)
    assert not hasattr(s, "mark_output")  # there is no output mutator at all


def test_sequential_compiles_from_one_store_are_independent() -> None:
    s, a, b = _two_analyses()
    first = bytes(s.reduce(outputs=[a])[0].serialize())
    second = bytes(s.reduce(outputs=[b])[0].serialize())
    again = bytes(s.reduce(outputs=[a])[0].serialize())
    assert first == _fresh_request_bytes(pick_second=False)  # history-independent
    assert second == _fresh_request_bytes(pick_second=True)
    assert again == first  # and no accumulation in either direction
    assert s.outputs() == []  # compiling never wrote store state


def test_serialize_outputs_scopes_the_unoptimized_bytes() -> None:
    s, _a, b = _two_analyses()
    scoped = gc.GraphStore.deserialize(bytes(s.serialize(outputs=[b])))
    assert _flagged(scoped) == [b]  # exactly the requested set
    bare = gc.GraphStore.deserialize(bytes(s.serialize()))
    assert _flagged(bare) == []  # an unreduced store has no outputs of its own
    # a deserialized artifact reserializes ITS carried outputs by default (the durable form)
    assert bytes(scoped.serialize()) == bytes(s.serialize(outputs=[b]))


def test_finalize_outputs_matches_one_shot_reduce() -> None:
    s, _a, b = _two_analyses()
    r = IncrementalReducer()
    inc_bytes = bytes(r.finalize(s, outputs=[b])[0].serialize())
    assert inc_bytes == bytes(s.reduce(outputs=[b])[0].serialize())
    assert inc_bytes == _fresh_request_bytes(pick_second=True)


def test_explicit_multi_output_request_keeps_both_and_is_deterministic() -> None:
    s, a, b = _two_analyses()
    reduced, _ = s.reduce(outputs=[a, b])
    assert len(_flagged(reduced)) == 2  # a deliberate multi-output compile still works
    assert bytes(s.reduce(outputs=[a, b])[0].serialize()) == bytes(reduced.serialize())


def test_invalid_output_ids_are_rejected() -> None:
    s, _a, _b = _two_analyses()
    with pytest.raises((ValueError, IndexError, OverflowError)):
        s.reduce(outputs=[10_000])
    with pytest.raises((ValueError, IndexError, OverflowError)):
        s.serialize(outputs=[10_000])


def test_concurrent_output_scoped_compiles_are_isolated() -> None:
    # compiling is READ-ONLY: many threads compiling DIFFERENT outputs from one shared store
    # must each get exactly their own single-output bytes (runs under 3.14t in CI)
    s, a, b = _two_analyses()
    want = {a: bytes(s.reduce(outputs=[a])[0].serialize()), b: bytes(s.reduce(outputs=[b])[0].serialize())}
    results: list[tuple[int, bytes]] = []
    lock = threading.Lock()

    def compile_one(node: int) -> None:
        got = bytes(s.reduce(outputs=[node])[0].serialize())
        with lock:
            results.append((node, got))

    threads = [threading.Thread(target=compile_one, args=(n,)) for n in (a, b) * 8]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 16
    assert all(got == want[node] for node, got in results)
