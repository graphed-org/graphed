"""graphed: the deferred-array recording surface.

Array expressions record into ``graphed.core`` instead of running, carrying the result form and
the user's source line with them; a backend (``graphed.awkward`` for ragged data,
``graphed.numpy`` for flat) supplies form inference and evaluation.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING

from .aggregate import aggregate_plan, collate, raise_on_floating_point
from .array import Array, apply
from .errors import (
    GraphedError,
    GraphedTypeError,
    GraphedWarning,
    OutputTypeError,
    PointError,
    WorkerLostError,
)
from .execute import compile_ir, evaluate_ir
from .materialization import PlanPieces, head, materialize
from .projection import read_columns
from .session import (
    Session,
    current_session,
    new_session,
    set_default_backend,
    set_session,
)
from .shuffle import join, join_plan, repartition, shuffle_plan
from .varied import SURFACE_DISPOSITIONS as SURFACE_DISPOSITIONS
from .varied import Varied, expand
from .vary import vary

#: Names that left the top level in 0.0.11, mapped to their home; each still resolves here, with a
#: DeprecationWarning, until 0.0.12.
_MOVED: dict[str, str] = {
    **dict.fromkeys(
        (
            "Explanation",
            "Kind",
            "broadcast_like",
            "context_of",
            "explain",
            "explain_variations",
            "impact_by_label",
            "labels",
            "member_of",
            "nominal",
            "points",
            "read_columns_by_label",
            "reindex_to",
            "selection",
            "unify_contexts",
            "universe",
            "variations",
            "weight",
        ),
        "graphed.systematics",
    ),
    "broadcasting": "graphed.systematics.varied",
    "expanding": "graphed.systematics.varied",
    **dict.fromkeys(("Provenance", "capture", "is_enabled", "set_enabled"), "graphed.provenance"),
    **dict.fromkeys(
        (
            "CONSERVATIVE",
            "BufferNeed",
            "BufferProjection",
            "OnFail",
            "Projection",
            "ProjectionError",
            "handle_opaque",
        ),
        "graphed.projection",
    ),
    **dict.fromkeys(("Backend", "Form", "ParamValue"), "graphed.backend"),
    **dict.fromkeys(("CompiledGraph", "refuse_chunk_partials"), "graphed.execute"),
    "resolve_backend": "graphed.aggregate",
    "BoundMethod": "graphed.array",
    "pack_key": "graphed.shuffle",
}

if TYPE_CHECKING:
    from .aggregate import resolve_backend as resolve_backend
    from .array import BoundMethod as BoundMethod
    from .backend import Backend as Backend
    from .backend import Form as Form
    from .backend import ParamValue as ParamValue
    from .execute import CompiledGraph as CompiledGraph
    from .execute import refuse_chunk_partials as refuse_chunk_partials
    from .projection import (
        CONSERVATIVE as CONSERVATIVE,
    )
    from .projection import (
        BufferNeed as BufferNeed,
    )
    from .projection import (
        BufferProjection as BufferProjection,
    )
    from .projection import (
        OnFail as OnFail,
    )
    from .projection import (
        Projection as Projection,
    )
    from .projection import (
        ProjectionError as ProjectionError,
    )
    from .projection import (
        handle_opaque as handle_opaque,
    )
    from .provenance import (
        Provenance as Provenance,
    )
    from .provenance import (
        capture as capture,
    )
    from .provenance import (
        is_enabled as is_enabled,
    )
    from .provenance import (
        set_enabled as set_enabled,
    )
    from .shuffle import pack_key as pack_key
    from .systematics import (
        Explanation as Explanation,
    )
    from .systematics import (
        Kind as Kind,
    )
    from .systematics import (
        broadcast_like as broadcast_like,
    )
    from .systematics import (
        context_of as context_of,
    )
    from .systematics import (
        explain as explain,
    )
    from .systematics import (
        explain_variations as explain_variations,
    )
    from .systematics import (
        impact_by_label as impact_by_label,
    )
    from .systematics import (
        labels as labels,
    )
    from .systematics import (
        member_of as member_of,
    )
    from .systematics import (
        nominal as nominal,
    )
    from .systematics import (
        points as points,
    )
    from .systematics import (
        read_columns_by_label as read_columns_by_label,
    )
    from .systematics import (
        reindex_to as reindex_to,
    )
    from .systematics import (
        selection as selection,
    )
    from .systematics import (
        unify_contexts as unify_contexts,
    )
    from .systematics import (
        universe as universe,
    )
    from .systematics import (
        variations as variations,
    )
    from .systematics import (
        weight as weight,
    )
    from .systematics.varied import broadcasting as broadcasting
    from .systematics.varied import expanding as expanding


def __getattr__(name: str) -> object:
    home = _MOVED.get(name)
    if home is None:
        raise AttributeError(f"module 'graphed' has no attribute {name!r}")
    warnings.warn(
        f"graphed.{name} moved to {home}.{name}; the top-level name is removed in 0.0.12",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(home), name)


def __dir__() -> list[str]:
    return __all__


#: How each public `Array`-consuming module verb treats a `Varied` container: expand it per label,
#: broadcast against it, read its metadata eagerly, or refuse it. `vary` is absent by construction
#: — it PRODUCES containers — and `evaluate_ir`/`unify_contexts` never take an `Array`.
VERB_DISPOSITIONS: dict[str, str] = {
    "aggregate_plan": "refusing",
    "apply": "expanding",
    "broadcast_like": "broadcasting",
    "compile_ir": "refusing",
    "context_of": "eager-metadata",
    "head": "refusing",
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
    "Array",
    "GraphedError",
    "GraphedTypeError",
    "GraphedWarning",
    "OutputTypeError",
    "PlanPieces",
    "PointError",
    "Session",
    "Varied",
    "WorkerLostError",
    "aggregate_plan",
    "apply",
    "collate",
    "compile_ir",
    "current_session",
    "evaluate_ir",
    "expand",
    "head",
    "join",
    "join_plan",
    "materialize",
    "new_session",
    "raise_on_floating_point",
    "read_columns",
    "repartition",
    "set_default_backend",
    "set_session",
    "shuffle_plan",
    "vary",
]

__version__ = "0.0.10"
