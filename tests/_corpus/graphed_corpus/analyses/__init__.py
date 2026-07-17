"""Canonical analyses (M0.5): the runnable fixtures with stored reference outputs."""

from __future__ import annotations

from .adl import ADL_QUERIES
from .systematics import TTBAR_FIXTURES, TTGAMMA_FIXTURES, ttbar_region, ttgamma_region

__all__ = ["ADL_QUERIES", "TTBAR_FIXTURES", "TTGAMMA_FIXTURES", "ttbar_region", "ttgamma_region"]
