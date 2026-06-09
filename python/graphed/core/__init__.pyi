"""Type stubs for the graphed-core package surface."""

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

# the compiled extension surface
from .graphed_core import (
    GraphStore as GraphStore,
)
from .graphed_core import (
    IncrementalReducer as IncrementalReducer,
)
from .graphed_core import (
    Params as Params,
)
from .graphed_core import (
    ParamValue as ParamValue,
)
from .graphed_core import (
    PayloadDescriptor as PayloadDescriptor,
)
from .graphed_core import (
    version as version,
)

# M8 durable plan (pure-Python)
from .plan import (
    Dataset as Dataset,
)
from .plan import (
    DurablePlan as DurablePlan,
)
from .plan import (
    OpSpec as OpSpec,
)
from .plan import (
    partition_dataset as partition_dataset,
)
from .plan import (
    partition_datasets as partition_datasets,
)
