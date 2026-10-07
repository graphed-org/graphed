"""``put`` from a fresh instance rewrites a blob whose size is wrong, on both store classes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from graphed.checkpoint import FsspecStore, Store

_DATA = b"hello world" * 100

_STORES: list[Callable[[Path], Any]] = [Store, lambda root: FsspecStore(root.as_uri())]


@pytest.mark.parametrize("make", _STORES, ids=["Store", "FsspecStore"])
def test_fresh_put_rewrites_a_truncated_blob(tmp_path: Path, make: Callable[[Path], Any]) -> None:
    digest = make(tmp_path).put(_DATA)
    (tmp_path / "objects" / digest).write_bytes(_DATA[:20])
    fresh = make(tmp_path)
    assert fresh.get(digest) is None
    fresh = make(tmp_path)
    assert fresh.put(_DATA) == digest
    assert fresh.get(digest) == _DATA
