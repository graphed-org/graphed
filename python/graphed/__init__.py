"""graphed: the deferred-array recording surface.

Array expressions record into ``graphed.core`` instead of running, carrying the result form and
the user's source line with them; a backend (``graphed.awkward`` for ragged data,
``graphed.numpy`` for flat) supplies form inference and evaluation.
"""

from __future__ import annotations

from .accessors import (
    broadcast_like,
    context_of,
    labels,
    nominal,
    points,
    reindex_to,
    selection,
    unify_contexts,
    universe,
    variations,
    weight,
)
from .aggregate import aggregate_plan, resolve_backend
from .array import Array, apply
from .backend import Backend, Form, ParamValue
from .by_label import impact_by_label, read_columns_by_label
from .errors import GraphedError, GraphedTypeError, VariationError
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
from .shuffle import join, join_plan, pack_key, repartition, shuffle_plan
from .varied import SURFACE_DISPOSITIONS, Varied, broadcasting, expanding, member_of
from .vary import vary

#: How each public `Array`-consuming module verb treats a `Varied` container: expand it per label,
#: broadcast against it, read its metadata eagerly, or refuse it. `vary` is absent by construction —
#: it PRODUCES containers — and `evaluate_ir`/`unify_contexts` never take an `Array`.
VERB_DISPOSITIONS: dict[str, str] = {
    "aggregate_plan": "refusing",
    "apply": "expanding",
    "broadcast_like": "broadcasting",
    "compile_ir": "refusing",
    "context_of": "eager-metadata",
    "impact_by_label": "expanding",
    "join": "refusing",
    "join_plan": "refusing",
    "pack_key": "refusing",
    "read_columns": "expanding",
    "read_columns_by_label": "expanding",
    "reindex_to": "broadcasting",
    "selection": "eager-metadata",
    "repartition": "refusing",
    "shuffle_plan": "refusing",
}

__all__ = [
    "CONSERVATIVE",
    "SURFACE_DISPOSITIONS",
    "VERB_DISPOSITIONS",
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
    "VariationError",
    "Varied",
    "aggregate_plan",
    "apply",
    "broadcast_like",
    "broadcasting",
    "capture",
    "compile_ir",
    "context_of",
    "evaluate_ir",
    "expanding",
    "handle_opaque",
    "impact_by_label",
    "is_enabled",
    "join",
    "join_plan",
    "labels",
    "member_of",
    "nominal",
    "pack_key",
    "points",
    "read_columns",
    "read_columns_by_label",
    "reindex_to",
    "repartition",
    "resolve_backend",
    "selection",
    "set_enabled",
    "shuffle_plan",
    "unify_contexts",
    "universe",
    "variations",
    "vary",
    "weight",
]

__version__ = "0.0.1"
