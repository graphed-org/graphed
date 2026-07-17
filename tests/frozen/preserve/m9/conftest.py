"""Shared fixtures for the M9 preservation suite."""

from __future__ import annotations

# Set BEFORE any OpenMP-using library (numpy/torch/xgboost) is imported below: the optional ML-plugin
# tests load torch AND xgboost, which each vendor their own OpenMP runtime; coexisting in one process
# on macOS otherwise aborts/segfaults ("OMP: Error #15 ... libomp already initialized"). This is a
# framework-vs-framework conflict, not the plugin API; harmless when those frameworks are absent (CI).
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Any

import agc
import numpy as np
import pytest


@pytest.fixture(scope="session")
def agc_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, np.ndarray]:
    """One built AGC bundle (with its build-time reference histogram), reused by read-only tests."""
    root = tmp_path_factory.mktemp("agc_bundle")
    return agc.build_agc(root)
