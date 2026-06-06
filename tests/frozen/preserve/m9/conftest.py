"""Shared fixtures for the M9 preservation suite."""

from __future__ import annotations

from typing import Any

import agc
import numpy as np
import pytest


@pytest.fixture(scope="session")
def agc_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, np.ndarray]:
    """One built AGC bundle (with its build-time reference histogram), reused by read-only tests."""
    root = tmp_path_factory.mktemp("agc_bundle")
    return agc.build_agc(root)
