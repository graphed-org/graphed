"""A stand-in wrapping library for the m60 extra suite, registered by trailing-dot spelling.

Its own module name — no frozen leg registers this prefix, and `m60x_libx` beside it is the
sibling a whole-dotted-component rule must leave alone.
"""

from __future__ import annotations

from typing import Any


def select(array: Any) -> Any:
    """Record an op from inside the library; the node's provenance is the seam under test."""
    return array.session.record_op("select", [array])
