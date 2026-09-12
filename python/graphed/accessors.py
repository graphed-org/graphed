"""`graphed.accessors` moved to `graphed.systematics.accessors` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.accessors`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.accessors import (  # isort: skip
    Introspectable as Introspectable,
    _follow as _follow,
    _is_context as _is_context,
    _variation_axis_index as _variation_axis_index,
    broadcast_like as broadcast_like,
    context_of as context_of,
    labels as labels,
    nominal as nominal,
    points as points,
    reindex_to as reindex_to,
    selection as selection,
    unify_contexts as unify_contexts,
    universe as universe,
    variations as variations,
    weight as weight,
    with_context as with_context,
)
