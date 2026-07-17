"""M36 — the model-parsing content hashes are memoized (parsed once, not per call).

A model's content hash is computed at record time and re-verified at bundle-build time; each call
otherwise re-parses the model. `memoized_model_hash` caches the result by a cheap raw-bytes digest
so the same payload parses once — bounded FIFO, keyed per (domain, bytes), and cold in a fresh
process so the by-value validate subprocess still computes correctly. (Framework-free.)
"""

from __future__ import annotations

from graphed_preserve.externals._helpers import (
    _HASH_MEMO,
    _HASH_MEMO_CAP,
    memoized_model_hash,
)


def setup_function() -> None:
    _HASH_MEMO.clear()


def test_repeated_payload_is_computed_once() -> None:
    calls: list[bytes] = []

    def compute(payload: bytes) -> str:
        calls.append(payload)
        return "sha256:" + str(len(payload))

    a = memoized_model_hash("dom", b"model-bytes", compute)
    b = memoized_model_hash("dom", b"model-bytes", compute)
    assert a == b
    assert calls == [b"model-bytes"]  # the expensive compute ran exactly once


def test_distinct_payloads_recompute() -> None:
    calls: list[bytes] = []
    compute = lambda p: calls.append(p) or ("sha256:" + str(len(p)))  # noqa: E731
    h1 = memoized_model_hash("dom", b"aaa", compute)
    h2 = memoized_model_hash("dom", b"bbbb", compute)
    assert h1 != h2 and len(calls) == 2  # different content -> recomputed


def test_domain_separates_identical_bytes() -> None:
    seen: list[str] = []
    compute = lambda p: seen.append("x") or "sha256:const"  # noqa: E731
    memoized_model_hash("onnx", b"same", compute)
    memoized_model_hash("torch", b"same", compute)
    assert len(seen) == 2  # same bytes, different plugin domain -> not a cache hit


def test_cache_is_bounded() -> None:
    for i in range(_HASH_MEMO_CAP + 20):
        memoized_model_hash("dom", f"payload-{i}".encode(), lambda p: "sha256:" + p.decode())
    assert len(_HASH_MEMO) <= _HASH_MEMO_CAP
