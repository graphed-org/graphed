"""graphed-core: Rust+PyO3 thread-safe interned graph IR + the M7 execution contract.

Re-exports the compiled extension and the (pure-Python, data-only) execution-layer protocol. The
graph lives in Rust; this package MUST NOT import awkward.
"""

from __future__ import annotations

from .execution import (
    ExecContext,
    ExecResult,
    Executor,
    Partition,
    Plan,
    StopCondition,
    StopReason,
    Task,
    WorkerResources,
)
from .graphed_core import GraphStore, PayloadDescriptor, version
from .plan import DurablePlan, OpSpec

__all__ = [
    "DurablePlan",
    "ExecContext",
    "ExecResult",
    "Executor",
    "GraphStore",
    "OpSpec",
    "Partition",
    "PayloadDescriptor",
    "Plan",
    "StopCondition",
    "StopReason",
    "Task",
    "WorkerResources",
    "version",
]
