"""A Session's equal payloads are one object, held no longer than it; a duplicate is never held."""

from __future__ import annotations

import gc
import json
import sys
import threading
from collections import OrderedDict
from typing import Any, cast

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.preserve.externals import _helpers
from graphed.preserve.externals._helpers import _payload_bytes
from graphed.preserve.externals.correctionlib_external import (
    _correctionlib_content_hash_impl,
    correctionlib_content_hash,
)

_SF = {"name": "sf", "inputs": [{"name": "x", "type": "real"}]}
CSET = json.dumps({"schema_version": 2, "corrections": [_SF]}).encode()


class _Counting(OrderedDict[Any, Any]):
    """Counts membership checks whose key equals a held key that carries another payload object."""

    content_compares = 0

    def _note(self, key: Any) -> None:
        if any(k == key and k[1] is not key[1] for k in list(self.keys())):
            self.content_compares += 1

    def __contains__(self, key: object) -> bool:
        self._note(key)
        return super().__contains__(key)


def _copy(data: bytes) -> bytes:
    duplicate = bytes(bytearray(data))
    assert duplicate == data and duplicate is not data
    return duplicate


def test_a_duplicate_gives_the_canonical_object_and_is_not_kept() -> None:
    s = Session(AwkwardBackend())
    first = b"payload-" * 1000
    duplicate = _copy(first)
    baseline = sys.getrefcount(duplicate)
    assert _payload_bytes(first, s) is first
    for _ in range(3):
        assert _payload_bytes(duplicate, s) is first
        assert _payload_bytes(cast("bytes", bytearray(first)), s) is first
    gc.collect()
    assert sys.getrefcount(duplicate) == baseline


def test_recording_a_duplicate_hashes_by_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    memo = _Counting()
    monkeypatch.setattr(_helpers, "_HASH_MEMO", memo)
    s = Session(AwkwardBackend())
    x = from_awkward("e", ak.zip({"x": np.array([0.5, 2.0])}, depth_limit=1), session=s).x
    gak.apply_correction(CSET, "sf", x=x)
    duplicate = _copy(CSET)
    for k in range(5):
        gak.apply_correction(duplicate, "sf", x=x + k)
    assert memo.content_compares == 0


def test_the_table_dies_with_its_session() -> None:
    s = Session(AwkwardBackend())
    first = _copy(b"set" * 100)  # a folded constant is immortal on free-threaded builds
    baseline = sys.getrefcount(first)
    _payload_bytes(first, s)
    assert sys.getrefcount(first) > baseline
    del s
    gc.collect()
    assert sys.getrefcount(first) == baseline


def test_sessions_do_not_share_payload_objects() -> None:
    a, b = Session(AwkwardBackend()), Session(AwkwardBackend())
    first, other = b"p" * 100, _copy(b"p" * 100)
    assert _payload_bytes(first, a) is first
    assert _payload_bytes(other, b) is other


def _recorded(payload: Any) -> Session:
    s = Session(AwkwardBackend())
    x = from_awkward("e", ak.zip({"x": np.array([0.5, 2.0])}, depth_limit=1), session=s).x
    gak.apply_correction(payload, "sf", x=x)
    return s


def test_a_dead_sessions_payload_leaves_the_hash_memo() -> None:
    payload = _copy(CSET)
    baseline = sys.getrefcount(payload)
    s = _recorded(payload)
    assert any(key[1] is payload for key in _helpers._HASH_MEMO)
    del s
    gc.collect()
    assert not any(key[1] is payload for key in _helpers._HASH_MEMO)
    assert sys.getrefcount(payload) == baseline


def test_a_dead_sessions_payload_file_leaves_the_read_cache(tmp_path: Any) -> None:
    path = tmp_path / "sf.json"
    path.write_bytes(CSET)
    gc.collect()  # an earlier test's dead Session must not clear the cache mid-test
    _helpers._read_payload.cache_clear()
    s = _recorded(str(path))
    assert _helpers._read_payload.cache_info().currsize == 1
    del s
    gc.collect()
    assert _helpers._read_payload.cache_info().currsize == 0


def test_a_live_session_hashes_its_equal_payload_after_another_dies() -> None:
    first, second = _copy(CSET), _copy(CSET)
    dying, live = _recorded(first), _recorded(second)
    del dying
    gc.collect()
    assert not any(key[1] == CSET for key in _helpers._HASH_MEMO)
    assert correctionlib_content_hash(second) == _correctionlib_content_hash_impl(CSET)
    assert any(key[1] is second for key in _helpers._HASH_MEMO)
    assert live._payloads[CSET] is second


def test_a_session_collected_under_the_memo_lock_is_purged_by_the_holder() -> None:
    payload = _copy(CSET)
    gc.disable()
    try:
        cycle: list[Any] = [_recorded(payload)]
        cycle.append(cycle)  # only a collection frees the Session
        del cycle

        def collect() -> None:
            with _helpers._HASH_MEMO_LOCK:
                gc.collect()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        t.join(10)
    finally:
        gc.enable()
    assert not t.is_alive()
    assert any(key[1] is payload for key in _helpers._HASH_MEMO)
    _helpers.memoized_model_hash("probe", b"other", lambda p: "v")
    assert not any(key[1] is payload for key in _helpers._HASH_MEMO)
