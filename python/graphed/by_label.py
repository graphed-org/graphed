"""`graphed.by_label` moved to `graphed.systematics.by_label` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.by_label`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.by_label import (  # isort: skip
    Outputs as Outputs,
    _is_sequence as _is_sequence,
    _per_label as _per_label,
    _reach as _reach,
    cone as cone,
    impact_by_label as impact_by_label,
    read_columns_by_label as read_columns_by_label,
)
