"""Type stubs for the graphed-core extension."""

from __future__ import annotations

# M7 execution contract (pure-Python; re-exported so downstream type-checkers see it)
from .execution import (
    ExecContext as ExecContext,
)
from .execution import (
    ExecResult as ExecResult,
)
from .execution import (
    Executor as Executor,
)
from .execution import (
    Partition as Partition,
)
from .execution import (
    Plan as Plan,
)
from .execution import (
    StopCondition as StopCondition,
)
from .execution import (
    StopReason as StopReason,
)
from .execution import (
    Task as Task,
)
from .execution import (
    WorkerResources as WorkerResources,
)

ParamValue = int | float | bool | str
Params = dict[str, ParamValue]

class PayloadDescriptor:
    """Reproducibility metadata an External node carries (participates in the structural hash)."""

    def __init__(
        self,
        *,
        kind: str,
        content_hash: str,
        framework: str,
        version: str,
        io_schema: str,
        preprocessing_ref: str | None = ...,
    ) -> None: ...
    @property
    def kind(self) -> str: ...
    @property
    def content_hash(self) -> str: ...
    @property
    def framework(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def io_schema(self) -> str: ...
    @property
    def preprocessing_ref(self) -> str | None: ...

class GraphStore:
    """Thread-safe interned graph store. Structurally identical nodes share one NodeId."""

    def __init__(self) -> None: ...
    def add_source(self, name: str, params: Params | None = ...) -> int: ...
    def add_op(self, name: str, inputs: list[int], params: Params | None = ...) -> int: ...
    def add_reduction(self, name: str, inputs: list[int], params: Params | None = ...) -> int: ...
    def add_external(
        self,
        descriptor: PayloadDescriptor | dict[str, str | None],
        inputs: list[int],
        params: Params | None = ...,
    ) -> int: ...
    def mark_output(self, node_id: int) -> None: ...
    def node_count(self) -> int: ...
    def to_dot(self) -> str: ...
    # M4 optimizer (DCE + CSE + equality-saturation stage fusion behind RewriteEngine):
    def reduce(self) -> tuple[GraphStore, dict[str, int]]: ...
    def reduce_incremental(self) -> tuple[GraphStore, dict[str, int]]: ...
    def reduction_report(self) -> dict[str, int]: ...

def version() -> str: ...
