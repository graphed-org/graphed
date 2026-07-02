"""graphed: deferred-array frontend that records a backend-agnostic program into graphed-core.

No fusion (that is M4). The awkward backend lives in graphed-awkward (M3). Provenance is real (M3).
"""

from __future__ import annotations

from .aggregate import aggregate_plan, resolve_backend
from .array import Array, apply
from .backend import Backend, Form, ParamValue
from .errors import GraphedError, GraphedTypeError
from .execute import CompiledGraph, compile_ir, evaluate_ir
from .projection import (
    CONSERVATIVE,
    BufferNeed,
    BufferProjection,
    OnFail,
    Projection,
    ProjectionError,
    handle_opaque,
    read_columns,
)
from .provenance import Provenance, capture, is_enabled, set_enabled
from .session import Session
from .shuffle import repartition, shuffle_plan

__all__ = [
    "CONSERVATIVE",
    "Array",
    "Backend",
    "BufferNeed",
    "BufferProjection",
    "CompiledGraph",
    "Form",
    "GraphedError",
    "GraphedTypeError",
    "OnFail",
    "ParamValue",
    "Projection",
    "ProjectionError",
    "Provenance",
    "Session",
    "aggregate_plan",
    "apply",
    "capture",
    "compile_ir",
    "evaluate_ir",
    "handle_opaque",
    "is_enabled",
    "read_columns",
    "repartition",
    "resolve_backend",
    "set_enabled",
    "shuffle_plan",
]

__version__ = "0.0.1"
