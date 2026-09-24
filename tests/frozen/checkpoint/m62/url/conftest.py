"""m62 unit B URL fixtures; unit C's s3 conftest defines the same two names."""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture(params=["memory", "file"])
def store_url(request, tmp_path) -> str:
    if request.param == "memory":
        return f"memory://m62-{uuid.uuid4().hex}"
    return "file://" + (tmp_path / "store").as_posix()


@pytest.fixture
def shared_url(tmp_path) -> str:
    return "file://" + (tmp_path / "store").as_posix()
