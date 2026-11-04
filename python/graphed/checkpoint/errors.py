"""Error harvesting — turning a failed partition into a reproducible dead-letter descriptor (M8).

When a partition fails, the runner records a **dead-letter descriptor**: enough plain data to find
and reproduce the failure. If the failure is a ``graphed.debug.StageError`` (the M6 source-mapped
error), its provenance — the user analysis line, the failing op, the input forms — is captured too,
so the dead letter points at the user's code, not an opaque worker string (plan A.3 #8).
"""

from __future__ import annotations

from graphed.core import dead_letter_descriptor

__all__ = ["dead_letter_descriptor"]
