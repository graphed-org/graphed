"""Error harvesting — turning a failed partition into a reproducible dead-letter descriptor (M8).

When a partition fails, the runner records a **dead-letter descriptor**: enough plain data to find
and reproduce the failure. If the failure is a ``graphed.debug.StageError`` (the M6 source-mapped
error), its provenance — the user analysis line, the failing op, the input forms — is captured too,
so the dead letter points at the user's code, not an opaque worker string (plan A.3 #8).
"""

from __future__ import annotations

from typing import Any

from graphed.core import Partition


def dead_letter_descriptor(task_id: str, partition: Partition, exc: BaseException) -> dict[str, Any]:
    """Build a reproducible, JSON-serializable dead-letter record from a failure."""
    desc: dict[str, Any] = {
        "task_id": task_id,
        "uri": partition.uri,
        "tree": partition.tree,
        "entry_start": partition.entry_start,
        "entry_stop": partition.entry_stop,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    # StageError carries structured user-source provenance (duck-typed so graphed-debug stays a
    # soft dependency: any error exposing these fields is harvested with full provenance).
    frames = getattr(exc, "frames", None)
    op = getattr(exc, "op", None)
    if frames is not None and op is not None:
        top = frames[0] if frames else None
        desc["stage_error"] = {
            "op": op,
            "cause_type": getattr(exc, "cause_type", ""),
            "cause_message": getattr(exc, "cause_message", ""),
            "user_file": getattr(top, "filename", "") if top else "",
            "user_line": getattr(top, "lineno", 0) if top else 0,
            "user_source": getattr(top, "source", "") if top else "",
        }
    return desc
