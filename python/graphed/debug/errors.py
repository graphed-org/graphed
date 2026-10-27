"""``StageError`` — the picklable, source-mapped error that survives a process boundary (plan M6).

A runtime failure inside a stage is wrapped in a ``StageError`` carrying the failing op, the **user
source-frame chain** (filename / line / function / sub-expression text), the input forms, and the
partition — all as plain data, so the error pickles intact and can be re-raised in the driver as
the same formatted traceback pointing at the user's analysis line. It never degrades to an opaque
string (plan A.3 #8). It deliberately does NOT carry the live Python traceback object (those do not
pickle); the source-frame chain is the durable, transportable record.

Both are defined in :mod:`graphed.errors`, below the frontend that raises them, and keep this
``graphed.debug`` name.
"""

from __future__ import annotations

from graphed.errors import SourceFrame, StageError

__all__ = ["SourceFrame", "StageError"]
