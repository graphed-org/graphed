"""A stand-in third-party library that records graphed ops on its caller's behalf.

A real importable module, not a stub: `capture()` classifies a frame by its module `__name__`, so
`m60_lib` and `m60_lib.sub` are what `register_internal("m60_lib")` must cover — and the sibling
`m60_libx` is what it must not.
"""

from __future__ import annotations

from typing import Any


def q(value: Any) -> Any:
    """A module-level callable named `q` — half of the same-name-different-module pair."""
    return value + 1


def select(array: Any) -> Any:
    """Record an op from inside the library; the node's provenance is the seam under test."""
    return array.session.record_op("select", [array])
