"""Coverage ratchet: ``_base.py`` validation/dispatch branches the frozen suites never take.

The frozen M9 suite proves ``validate_plugin`` rejects a vacuous or process-nondeterministic hash;
it never tries too-few or duplicate samples, or a hash function that returns an empty string. And
every frozen call into ``evaluate_external`` goes through a ``ResourceCache`` (the ``bundle.py``
path) — the uncached load/evaluate/close shot is never exercised.
"""

from __future__ import annotations

from typing import Any

import cloudpickle
import pytest

from graphed.preserve import (
    ExternalPlugin,
    PreserveError,
    evaluate_external,
    register_plugin,
    registered_kinds,
    validate_plugin,
)
from graphed.preserve.externals._base import _hash_in_subprocess


def test_registered_kinds_lists_the_builtins_sorted() -> None:
    kinds = registered_kinds()
    assert kinds == sorted(kinds)
    assert {"correctionlib", "triton_model", "onnx_model"} <= set(kinds)


def _two_samples() -> list[bytes]:
    return [b"alpha", b"beta"]


def test_validate_plugin_rejects_fewer_than_two_samples() -> None:
    plugin = ExternalPlugin(
        kind="single-sample",
        content_hash=lambda p: "sha256:x",
        evaluate=lambda *_: None,
        samples=lambda: [b"only"],
    )
    with pytest.raises(PreserveError, match=">=2 distinct payloads"):
        validate_plugin(plugin)


def test_validate_plugin_rejects_duplicate_samples() -> None:
    plugin = ExternalPlugin(
        kind="dup-sample",
        content_hash=lambda p: f"sha256:{p!r}",
        evaluate=lambda *_: None,
        samples=lambda: [b"same", b"same"],
    )
    with pytest.raises(PreserveError, match="must be distinct"):
        validate_plugin(plugin)


def test_validate_plugin_rejects_an_empty_hash() -> None:
    plugin = ExternalPlugin(
        kind="empty-hash", content_hash=lambda p: "", evaluate=lambda *_: None, samples=_two_samples
    )
    with pytest.raises(PreserveError, match="non-empty string"):
        validate_plugin(plugin)


def _module_hash(payload: bytes) -> str:
    return f"sha256:{payload.hex()}"


def test_hash_in_subprocess_tolerates_an_unresolvable_module() -> None:
    # a content_hash whose __module__ names nothing in sys.modules: register_pickle_by_value is
    # skipped entirely, and the hash still reproduces via plain (non-by-value) pickling.
    def orphan_hash(payload: bytes) -> str:
        return f"sha256:{payload.hex()}"

    orphan_hash.__module__ = "graphed_test_module_that_was_never_imported"
    got = _hash_in_subprocess(orphan_hash, b"alpha", seed="0")
    assert got == orphan_hash(b"alpha")


def test_hash_in_subprocess_surfaces_a_failed_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    # register_pickle_by_value failing forces the by-value path off; the hash fn then pickles by
    # reference to this (uninstalled, test-only) module, which the subprocess cannot import —
    # the failure must surface as PreserveError, not a bare CalledProcessError/traceback.
    def boom(module: Any) -> None:
        raise RuntimeError("cannot register this module")

    monkeypatch.setattr(cloudpickle, "register_pickle_by_value", boom)
    with pytest.raises(PreserveError, match="content_hash failed in a subprocess"):
        _hash_in_subprocess(_module_hash, b"alpha", seed="0")


def test_evaluate_external_without_a_cache_loads_evaluates_and_closes() -> None:
    calls: list[str] = []

    def load(payload: bytes, params: Any) -> bytes:
        calls.append("load")
        return payload

    def evaluate(resource: bytes, params: Any, inputs: list[Any]) -> Any:
        calls.append("evaluate")
        return resource + inputs[0]

    def close(resource: bytes) -> None:
        calls.append("close")

    plugin = ExternalPlugin(
        kind="no-cache",
        content_hash=lambda p: f"sha256:{p.hex()}",
        load=load,
        evaluate=evaluate,
        close=close,
        samples=_two_samples,
    )
    register_plugin(plugin, validate=False)
    node = {"descriptor": {"kind": "no-cache", "content_hash": "sha256:x"}, "params": {}}
    result = evaluate_external(node, [b"-input"], b"payload", cache=None)
    assert result == b"payload-input"
    assert calls == ["load", "evaluate", "close"]  # single-shot: no resource survives the call
