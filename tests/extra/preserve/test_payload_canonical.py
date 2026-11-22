"""#326: equal payload objects become one, so a duplicate's content is compared once."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, cast

import awkward as ak
import numpy as np
import pytest

from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.preserve.externals import _helpers
from graphed.preserve.externals._helpers import _payload_bytes

CSET = json.dumps(
    {
        "schema_version": 2,
        "corrections": [{"name": "sf", "inputs": [{"name": "x", "type": "real"}]}],
    }
).encode()


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


@pytest.fixture
def canon(monkeypatch: pytest.MonkeyPatch) -> _Counting:
    counting = _Counting()
    monkeypatch.setattr(_helpers, "_CANON", counting)
    monkeypatch.setattr(_helpers, "_SEEN", OrderedDict())
    return counting


def _copy(data: bytes) -> bytes:
    duplicate = bytes(bytearray(data))
    assert duplicate == data and duplicate is not data
    return duplicate


def test_a_duplicate_object_is_compared_once(canon: _Counting) -> None:
    first = b"payload-" * 1000
    duplicate = _copy(first)
    assert _payload_bytes(first) is first
    for _ in range(10):
        assert _payload_bytes(duplicate) is first
    assert canon.content_compares == 1


def test_recording_a_duplicate_hashes_by_identity(
    canon: _Counting, monkeypatch: pytest.MonkeyPatch
) -> None:
    memo = _Counting()
    monkeypatch.setattr(_helpers, "_HASH_MEMO", memo)
    s = Session(AwkwardBackend())
    x = from_awkward(s, "e", ak.zip({"x": np.array([0.5, 2.0])}, depth_limit=1)).x
    gak.apply_correction(CSET, "sf", x=x)
    duplicate = _copy(CSET)
    for k in range(5):
        gak.apply_correction(duplicate, "sf", x=x + k)
    assert memo.content_compares == 0
    assert canon.content_compares == 1


def test_a_bytearray_is_canonical_but_not_kept_by_id(canon: _Counting) -> None:
    first = b"raw" * 100
    _payload_bytes(first)
    for _ in range(3):
        assert _payload_bytes(cast("bytes", bytearray(first))) is first
    assert len(_helpers._SEEN) == 0


def test_the_tables_are_bounded(canon: _Counting) -> None:
    for i in range(_helpers._CANON_CAP + 5):
        first = f"p{i}".encode() * 50
        _payload_bytes(first)
        _payload_bytes(_copy(first))
    assert len(_helpers._CANON) == len(_helpers._SEEN) == _helpers._CANON_CAP
