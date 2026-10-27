"""`graphed._tags` moved to `graphed.systematics.tags` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.tags`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.tags import (  # isort: skip
    MAX_TAG_CHARS as MAX_TAG_CHARS,
    canonical_tag as canonical_tag,
    numeric_value as numeric_value,
    python_number as python_number,
)
