"""graphed: deferred-array frontend that records a backend-agnostic program into graphed-core.

No fusion (that is M4). The awkward backend lives in graphed-awkward (M3). Provenance is real (M3).
"""

from __future__ import annotations

from .array import Array
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
)
from .provenance import Provenance, capture, is_enabled, set_enabled
from .session import Session

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
    "capture",
    "compile_ir",
    "evaluate_ir",
    "handle_opaque",
    "is_enabled",
    "set_enabled",
]

__version__ = "0.0.1"
