"""A ``dir::`` store URL roots the store at the wrapped URL, with its options on that filesystem."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from graphed.checkpoint import FsspecStore

_DIR_WARNING = "pass the wrapped URL instead"


def test_dir_store_lands_at_the_wrapped_root(tmp_path: Path) -> None:
    root = tmp_path / "ck"
    with pytest.warns(UserWarning, match=_DIR_WARNING):
        digest = FsspecStore(f"dir::{root.as_uri()}").put(b"blob")
    assert (root / "objects" / digest).read_bytes() == b"blob"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert FsspecStore(root.as_uri()).get(digest) == b"blob"


def test_dir_s3_store_takes_its_endpoint_option(s3_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    s3fs = pytest.importorskip("s3fs")
    endpoint = os.environ["AWS_ENDPOINT_URL"]
    # only the option names the live endpoint: nothing listens on the discard port
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    s3fs.S3FileSystem.clear_instance_cache()
    with pytest.warns(UserWarning, match=_DIR_WARNING):
        store = FsspecStore(f"dir::{s3_url}", client_kwargs={"endpoint_url": endpoint})
    digest = store.put(b"blob")
    assert FsspecStore(s3_url, client_kwargs={"endpoint_url": endpoint}).get(digest) == b"blob"
