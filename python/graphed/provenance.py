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

#: Module-name prefixes `capture` skips, each already dot-terminated except the built-in
#: `graphed*` rule, which is a bare STRING prefix (`graphed_foo` counts as internal, as it always
#: has). A tuple so the per-frame test in `capture` — on the hot path of every recorded op — stays
#: ONE `str.startswith` call however many libraries have registered.
_SKIP: tuple[str, ...] = ("graphed",)


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


def register_internal(prefix: str) -> None:
    """Declare ``prefix`` a wrapping library, so the ops it records point at ITS caller's line.

    A library that records graphed ops on a user's behalf is not the user: without this every node
    it records maps to library source instead of the analysis line. ``prefix`` matches whole dotted
    components — ``"lib"`` covers ``lib`` and ``lib.sub``, never ``libx``; ``"lib."`` is the same
    declaration, spelled with the separator. Idempotent, thread-safe and process-global, with no
    inverse: a library registers itself once, at import.

    :raises ValueError: if any dotted component of ``prefix`` — a trailing separator aside — is
        not an identifier (``"a..b"`` and ``"my lib"`` are two such). No module name can equal such
        a prefix, so registering it would leave in place exactly the wrong provenance this call
        fixes.
    """
    global _SKIP
    prefix = prefix.rstrip(".")
    if not all(part.isidentifier() for part in prefix.split(".")):
        raise ValueError(f"graphed: register_internal needs a module prefix, got {prefix!r}")
    with _lock:  # through a set, so a repeat registration cannot grow the per-frame test
        _SKIP = tuple(sorted({*_SKIP, f"{prefix}."}))


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
    skip = _SKIP  # one read of the rebound global, so a concurrent registration cannot split a walk
    while frame is not None:
        # the trailing dot is what makes a registered prefix match whole dotted COMPONENTS; it
        # cannot change the built-in `graphed` answer, which is a prefix of the name either way
        if not (frame.f_globals.get("__name__", "") + ".").startswith(skip):
            return Provenance(
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                function=frame.f_code.co_name,
                source=_source_text(frame),
            )
        frame = frame.f_back
    return Provenance("<unknown>", 0)
