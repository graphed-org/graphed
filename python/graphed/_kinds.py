"""`graphed._kinds` moved to `graphed.systematics.kinds` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.kinds`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.kinds import (  # isort: skip
    Kind as Kind,
)
