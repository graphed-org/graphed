"""m62 unit B URL fixtures; unit C's s3 conftest defines the same two names."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=["memory", "file"])
def store_url(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    if request.param == "memory":
        return f"memory://m62-{uuid.uuid4().hex}"
    return "file://" + (tmp_path / "store").as_posix()


@pytest.fixture
def shared_url(tmp_path: Path) -> str:
    return "file://" + (tmp_path / "store").as_posix()
