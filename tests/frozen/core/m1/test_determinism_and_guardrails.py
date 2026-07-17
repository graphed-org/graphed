"""to_dot determinism + M1 guardrails (graph lives in Rust; graphed-core must not import awkward)."""

from __future__ import annotations

import sys

import pytest

import graphed.core as gc


def _build(store: gc.GraphStore) -> None:
    src = store.add_source("events", {"uri": "f.root", "tree": "Events"})
    pt = store.add_op("pt", [src])
    cut = store.add_op("cut", [pt], {"thr": 30})
    store.add_reduction("sum", [cut])
    store.add_external(
        {
            "kind": "correctionlib",
            "content_hash": "sha256:1",
            "framework": "correctionlib",
            "version": "2.6",
            "io_schema": "json",
            "preprocessing_ref": None,
        },
        [cut],
    )


def test_to_dot_is_byte_stable() -> None:
    a = gc.GraphStore()
    b = gc.GraphStore()
    _build(a)
    _build(b)
    assert a.to_dot() == b.to_dot()
    # stable across repeated calls on the same store too
    assert a.to_dot() == a.to_dot()


def test_to_dot_is_nonempty_digraph() -> None:
    s = gc.GraphStore()
    _build(s)
    dot = s.to_dot()
    assert dot.startswith("digraph")
    assert "->" in dot


def test_version_returns_string() -> None:
    assert isinstance(gc.version(), str)
    assert gc.version()


def test_core_does_not_import_awkward() -> None:
    # Guardrail A.4: graphed-core MUST NOT depend on awkward.
    assert "awkward" not in sys.modules


def test_reduce_outputs_accepts_valid_and_rejects_invalid() -> None:
    # [freeze-M22-1, user-authorized respin: outputs are given per compile request — the
    # mark_output mutator is REMOVED from the public API]
    s = gc.GraphStore()
    n = s.add_source("e")
    s.reduce(outputs=[n])  # ok
    assert not hasattr(s, "mark_output")  # the mutator is gone
    with pytest.raises((ValueError, IndexError, OverflowError)):
        s.reduce(outputs=[10_000])  # out of range
