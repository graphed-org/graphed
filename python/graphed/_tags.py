"""`graphed._tags` moved to `graphed.systematics.tags` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.tags`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.tags import (  # isort: skip
    _CANONICAL as _CANONICAL,
    _FLOAT_SUGAR as _FLOAT_SUGAR,
    _IDENTIFIER as _IDENTIFIER,
    _NOT_FINITE as _NOT_FINITE,
    _P_FORM as _P_FORM,
    MAX_TAG_CHARS as MAX_TAG_CHARS,
    _normalize as _normalize,
    _render as _render,
    canonical_tag as canonical_tag,
    numeric_value as numeric_value,
    python_number as python_number,
)
