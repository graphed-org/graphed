"""graphed-core: Rust+PyO3 thread-safe interned graph IR + the M7 execution contract.

Re-exports the compiled extension and the (pure-Python, data-only) execution-layer protocol. The
graph lives in Rust; this package MUST NOT import awkward.
"""

from __future__ import annotations

from .execution import (
    AddressTable,
    ClusterExecutor,
    ExecContext,
    ExecResult,
    Executor,
    LocalResources,
    Monitor,
    NodeStore,
    Partition,
    Plan,
    RunControl,
    RunState,
    SequentialRunner,
    ShuffleBackend,
    StopCondition,
    StopReason,
    Task,
    TaskEvent,
    TaskPhase,
    WorkerProfiler,
    WorkerResources,
    emit_task,
    lean_events,
    partition_label,
    worker_monitor_factory,
)
from .graphed_core import GraphStore, IncrementalReducer, PayloadDescriptor, version
from .plan import (
    Dataset,
    DurablePlan,
    DurablePlanV2,
    OpSpec,
    StageSpec,
    partition_dataset,
    partition_datasets,
)

__all__ = [
    "AddressTable",
    "ClusterExecutor",
    "Dataset",
    "DurablePlan",
    "DurablePlanV2",
    "ExecContext",
    "ExecResult",
    "Executor",
    "GraphStore",
    "IncrementalReducer",
    "LocalResources",
    "Monitor",
    "NodeStore",
    "OpSpec",
    "Partition",
    "PayloadDescriptor",
    "Plan",
    "RunControl",
    "RunState",
    "SequentialRunner",
    "ShuffleBackend",
    "StageSpec",
    "StopCondition",
    "StopReason",
    "Task",
    "TaskEvent",
    "TaskPhase",
    "WorkerProfiler",
    "WorkerResources",
    "emit_task",
    "lean_events",
    "partition_dataset",
    "partition_datasets",
    "partition_label",
    "version",
    "worker_monitor_factory",
]
