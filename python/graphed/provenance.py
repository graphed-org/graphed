"""Provenance capture (real implementation, M3 — replaces the M2 stub).

Captures the first stack frame outside the ``graphed*`` packages: filename, line number, the
enclosing function, and the **sub-expression source text** (via ``executing``) so a node maps to the
exact piece of user code that created it. Capture is stateless and therefore thread-safe (it only
reads the calling thread's own stack). It is toggleable so builds can opt out of the overhead.
"""

from __future__ import annotations

import ast
import sys
import threading
from dataclasses import dataclass
from types import FrameType

try:  # executing gives sub-expression text; degrade gracefully if unavailable
    import executing
except Exception:  # pragma: no cover - executing is a declared dependency
    executing = None  # type: ignore[assignment]

_enabled = True
_lock = threading.Lock()


@dataclass(frozen=True)
class Provenance:
    filename: str
    lineno: int
    function: str = ""
    source: str = ""

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}"


_DISABLED = Provenance("<provenance-disabled>", 0)


def set_enabled(value: bool) -> None:
    """Globally toggle provenance capture (default on)."""
    global _enabled
    with _lock:
        _enabled = value


def is_enabled() -> bool:
    return _enabled


def _source_text(frame: FrameType) -> str:
    if executing is None:  # pragma: no cover
        return ""
    try:
        node = executing.Source.executing(frame).node
    except Exception:  # pragma: no cover - executing is best-effort
        return ""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return ""


def capture() -> Provenance:
    """Return provenance for the first stack frame outside the graphed* packages (the user's line).

    Walks the frame chain directly (`f_back`) rather than `inspect.stack()`, which builds a
    `FrameInfo` — reading and `stat`ing the source file — for EVERY frame before we discard all but
    the first user frame. `record_op` calls this on every op, so a systematics fanout pays it per
    universe; the direct walk stats no source and reads it only for the one frame it keeps (via
    `executing` in `_source_text`), the fields being lifted straight off the frame object.
    """
    if not _enabled:
        return _DISABLED
    frame: FrameType | None = sys._getframe(1)  # capture()'s caller; its own frame is graphed*
    while frame is not None:
        if not frame.f_globals.get("__name__", "").startswith("graphed"):
            return Provenance(
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                function=frame.f_code.co_name,
                source=_source_text(frame),
            )
        frame = frame.f_back
    return Provenance("<unknown>", 0)
