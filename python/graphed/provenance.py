"""Provenance capture (real implementation, M3 — replaces the M2 stub).

Captures the first stack frame outside the ``graphed*`` packages: filename, line number, the
enclosing function, and the **sub-expression source text** (via ``executing``) so a node maps to
the exact piece of user code that created it. Capture is stateless and therefore thread-safe (it
only reads the calling thread's own stack). It is toggleable so builds can opt out of the overhead.
"""

from __future__ import annotations

import ast
import os
import sys
import sysconfig
import threading
from dataclasses import dataclass
from types import FrameType

try:  # executing gives sub-expression text; degrade gracefully if unavailable
    import executing
except ImportError:  # pragma: no cover - executing is a declared dependency
    executing = None  # type: ignore[assignment]

_lock = threading.Lock()


@dataclass
class _Settings:
    """The process-global capture settings; each field is rebound whole, under `_lock`."""

    enabled: bool = True
    #: Module-name prefixes `capture` skips, each already dot-terminated except the built-in
    #: `graphed*` rule, which is a bare STRING prefix (`graphed_foo` counts as internal, as it
    #: always has). A tuple so the per-frame test in `capture` — on the hot path of every recorded
    #: op — stays ONE `str.startswith` call however many libraries have registered.
    skip: tuple[str, ...] = ("graphed",)


_settings = _Settings()

#: installed third-party code (site-packages), skipped unless no frame lies outside it: an analysis
#: installed non-editable lives there too, and its own frame is then the best answer left.
_INSTALLED: tuple[str, ...] = tuple(
    sorted(
        {
            os.path.join(path, "")
            for key in ("purelib", "platlib")
            for path in (sysconfig.get_path(key), os.path.realpath(sysconfig.get_path(key)))
        }
    )
)


@dataclass(frozen=True)
class Provenance:
    """A user source location: file, line, function and source text."""

    filename: str
    lineno: int
    function: str = ""
    source: str = ""

    def __str__(self) -> str:
        """Show the location as ``filename:lineno``."""
        return f"{self.filename}:{self.lineno}"


_DISABLED = Provenance("<provenance-disabled>", 0)


def set_enabled(value: bool) -> None:
    """Globally toggle provenance capture (default on)."""
    with _lock:
        _settings.enabled = value


def is_enabled() -> bool:
    """Return whether provenance capture is on."""
    return _settings.enabled


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
    prefix = prefix.rstrip(".")
    if not all(part.isidentifier() for part in prefix.split(".")):
        raise ValueError(f"graphed: register_internal needs a module prefix, got {prefix!r}")
    with _lock:  # through a set, so a repeat registration cannot grow the per-frame test
        _settings.skip = tuple(sorted({*_settings.skip, f"{prefix}."}))


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
    except Exception:  # pragma: no cover - best-effort, as above
        return ""


def capture() -> Provenance:
    """Return provenance for the user's line.

    The first stack frame outside the graphed* packages and outside installed third-party code,
    else the first frame outside graphed*.

    Walks the frame chain directly (`f_back`) rather than `inspect.stack()`, which builds a
    `FrameInfo` — reading and `stat`ing the source file — for EVERY frame before we discard all but
    the first user frame. `record_op` calls this on every op, so a systematics fanout pays it per
    universe; the direct walk stats no source and reads it only for the one frame it keeps (via
    `executing` in `_source_text`), the fields being lifted straight off the frame object.
    """
    if not _settings.enabled:
        return _DISABLED
    frame: FrameType | None = sys._getframe(1)  # capture()'s caller; its own frame is graphed*
    skip = _settings.skip  # one read, so a concurrent registration cannot split a walk
    first: FrameType | None = None
    while frame is not None:
        # the trailing dot is what makes a registered prefix match whole dotted COMPONENTS; it
        # cannot change the built-in `graphed` answer, which is a prefix of the name either way
        if not (frame.f_globals.get("__name__", "") + ".").startswith(skip):
            if not frame.f_code.co_filename.startswith(_INSTALLED):
                return _provenance(frame)
            first = first or frame
        frame = frame.f_back
    return Provenance("<unknown>", 0) if first is None else _provenance(first)


def _provenance(frame: FrameType) -> Provenance:
    return Provenance(
        filename=frame.f_code.co_filename,
        lineno=frame.f_lineno,
        function=frame.f_code.co_name,
        source=_source_text(frame),
    )
