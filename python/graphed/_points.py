"""`graphed._points` moved to `graphed.systematics.points` (m57); this re-exports it.

Every name is the same object as `graphed.systematics.points`'s, so an `isinstance` check or an
import through either path means exactly the same thing.
"""

from __future__ import annotations

from .systematics.points import (  # isort: skip
    Point as Point,
    _build as _build,
    _decimal as _decimal,
    _nuisance as _nuisance,
    coordinate as coordinate,
    default as default,
    render as render,
    restrict as restrict,
)
