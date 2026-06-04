"""graphed-core: Rust+PyO3 thread-safe interned graph IR.

Re-exports the compiled extension. The graph lives in Rust; this package MUST NOT import awkward.
"""

from __future__ import annotations

from .graphed_core import GraphStore, PayloadDescriptor, version

__all__ = ["GraphStore", "PayloadDescriptor", "version"]
