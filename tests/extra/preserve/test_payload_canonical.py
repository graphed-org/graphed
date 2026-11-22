"""#326/#328: a Session's equal payloads are one object, compared once, held no longer than it."""

from __future__ import annotations

import gc
import json
import sys
from collections import OrderedDict
from typing import Any, cast

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.preserve.externals import _helpers
from graphed.preserve.externals._helpers import _payload_bytes

_SF = {"name": "sf", "inputs": [{"name": "x", "type": "real"}]}
CSET = json.dumps({"schema_version": 2, "corrections": [_SF]}).encode()


class _Counting(OrderedDict[Any, Any]):
    """Counts lookups whose key equals a held key without being the same object."""

    content_compares = 0
    inside = False  # setdefault looks the key up again through the overrides below

    def _note(self, key: Any) -> None:
        if self.inside:
            return
        bare = key[1] if isinstance(key, tuple) else key
        held = [k[1] if isinstance(k, tuple) else k for k in list(self.keys()) if k == key]
        if any(h is not bare for h in held):
            self.content_compares += 1

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self._note(key)
        self.inside = True
        try:
            return super().setdefault(key, default)
        finally:
            self.inside = False

    def __contains__(self, key: object) -> bool:
        self._note(key)
        return super().__contains__(key)

    def __getitem__(self, key: Any) -> Any:
        self._note(key)
        return super().__getitem__(key)

    def move_to_end(self, key: Any, last: bool = True) -> None:
        self._note(key)
        super().move_to_end(key, last)


def _session() -> tuple[Session, _Counting]:
    s = Session(AwkwardBackend())
    s._payloads = counting = _Counting()
    return s, counting


def _copy(data: bytes) -> bytes:
    duplicate = bytes(bytearray(data))
    assert duplicate == data and duplicate is not data
    return duplicate


def test_a_duplicate_object_is_compared_once() -> None:
    s, canon = _session()
    first = b"payload-" * 1000
    duplicate = _copy(first)
    assert _payload_bytes(first, s) is first
    for _ in range(10):
        assert _payload_bytes(duplicate, s) is first
    assert canon.content_compares == 1


def test_recording_a_duplicate_hashes_by_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    memo = _Counting()
    monkeypatch.setattr(_helpers, "_HASH_MEMO", memo)
    s, canon = _session()
    x = from_awkward("e", ak.zip({"x": np.array([0.5, 2.0])}, depth_limit=1), session=s).x
    gak.apply_correction(CSET, "sf", x=x)
    duplicate = _copy(CSET)
    for k in range(5):
        gak.apply_correction(duplicate, "sf", x=x + k)
    assert memo.content_compares == 0
    assert canon.content_compares == 1


def test_a_bytearray_is_canonical_but_not_kept_by_id() -> None:
    s = Session(AwkwardBackend())
    first = b"raw" * 100
    _payload_bytes(first, s)
    for _ in range(3):
        assert _payload_bytes(cast("bytes", bytearray(first)), s) is first
    assert s._payload_ids == {}


def test_the_tables_die_with_their_session() -> None:
    s = Session(AwkwardBackend())
    _payload_bytes(b"set" * 100, s)
    duplicate = _copy(b"set" * 100)
    baseline = sys.getrefcount(duplicate)
    _payload_bytes(duplicate, s)
    assert sys.getrefcount(duplicate) > baseline
    del s
    gc.collect()
    assert sys.getrefcount(duplicate) == baseline


def test_sessions_do_not_share_payload_objects() -> None:
    a, b = Session(AwkwardBackend()), Session(AwkwardBackend())
    first, other = b"p" * 100, _copy(b"p" * 100)
    assert _payload_bytes(first, a) is first
    assert _payload_bytes(other, b) is other


def test_the_duplicates_a_session_keeps_are_bounded() -> None:
    s = Session(AwkwardBackend())
    for i in range(_helpers._PAYLOAD_IDS_CAP + 5):
        first = f"p{i}".encode() * 50
        _payload_bytes(first, s)
        _payload_bytes(_copy(first), s)
    assert len(s._payload_ids) == _helpers._PAYLOAD_IDS_CAP
