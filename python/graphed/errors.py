"""graphed frontend errors, and the worker-side ``StageError`` the frontend raises (plan M6)."""

from __future__ import annotations

import linecache
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .provenance import _INSTALLED, _STDLIB, Provenance, _settings


class GraphedError(Exception):
    """Base class for graphed frontend errors."""


class GraphedWarning(UserWarning):
    """Something recorded, but not as well as it could have been."""


class PointError(GraphedError):
    """A ``points=`` entry that does not fit the situation it names (design §4).

    ``situation`` discriminates the misuse — one of ``"unresolved" | "unreachable" | "duplicate" |
    "empty" | "conflict"`` — so a caller can ``except PointError`` and branch on ``.situation``
    without parsing the message. ``entry`` is the offending declare/placement and ``valid`` the set
    it should have named; ``detail`` is the human explanation the message carries.
    """

    def __init__(self, situation: str, entry: Any, *, valid: Any = None, detail: str = "") -> None:
        """Name the ``situation``, the offending ``entry`` and what would be ``valid``."""
        self.situation = situation
        self.entry = entry
        self.valid = valid
        self.detail = detail
        super().__init__(f"{situation}: {detail}" if detail else situation)


class GraphedTypeError(GraphedError):
    """An ill-typed op, raised at the user's source line (plan M2)."""

    def __init__(self, op: str, provenance: Provenance, detail: str = "") -> None:
        """Name the ill-typed ``op`` and the user line that recorded it."""
        self.op = op
        self.provenance = provenance
        self.detail = detail
        msg = f"ill-typed op {op!r} at {provenance}"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


class SourceRowsError(GraphedTypeError, TypeError):
    """An op pairing the rows of two partitioned sources, which need not be the same events."""


class OutputTypeError(GraphedTypeError):
    """An External's value disagrees with the type its node declared.

    Located at the declaring line.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by constructor arguments."""
        return (type(self), (self.op, self.provenance, self.detail))


class PackKeyOverflowError(ValueError):
    """A key field too wide for ``pack_key``: its rows would collide, so the run stops here."""


class MissingFieldError(GraphedTypeError, AttributeError):
    """``arr.<name>`` naming no field.

    An ill-typed ``field`` op that ``hasattr`` reads as absent.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by constructor arguments."""
        return (type(self), (self.op, self.provenance, self.detail))


@dataclass(frozen=True)
class SourceFrame:
    """One user-source frame: where a node was created in the user's analysis."""

    filename: str
    lineno: int
    function: str = ""
    source: str = ""
    #: the other lines that wrote this same expression, which recording merged into one node
    also: tuple[Provenance, ...] = ()

    @classmethod
    def of(cls, sites: Sequence[Provenance]) -> SourceFrame:
        """Return the first of a node's recording ``sites``, carrying the rest as ``also``."""
        first, *rest = sites
        return cls(first.filename, first.lineno, first.function, first.source, tuple(rest))

    def __str__(self) -> str:
        """Show the frame as ``filename:lineno``."""
        return f"{self.filename}:{self.lineno}"


def raising_frame(exc: BaseException) -> SourceFrame | None:
    """Return the deepest frame of ``exc``'s traceback that is the user's, or ``None``.

    An op failing inside a library has none: its traceback is all ``graphed``, third-party and
    standard-library frames.
    """
    skip = _settings.skip
    found = None
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        filename = frame.f_code.co_filename
        internal = (frame.f_globals.get("__name__", "") + ".").startswith(skip)
        if not internal and not filename.startswith((*_INSTALLED, _STDLIB, "<frozen")):
            found = (frame, lineno)
    if found is None:
        return None
    frame, lineno = found
    filename = frame.f_code.co_filename
    source = linecache.getline(filename, lineno, frame.f_globals).strip()
    return SourceFrame(filename, lineno, frame.f_code.co_name, source)


_NO_SOURCE_LINE = SourceFrame("<no source line: the failure was outside any recorded op>", 0)


class StageError(Exception):
    """A stage/op failure mapped back to the user's analysis. Picklable and re-raisable intact."""

    def __init__(
        self,
        *,
        op: str,
        frames: tuple[SourceFrame, ...],
        input_forms: tuple[str, ...],
        partition: str,
        cause_type: str,
        cause_message: str,
        opt_level: int,
        variation: str = "",
        raised_at: SourceFrame | None = None,
    ) -> None:
        """Describe a failed op: its user frames, input forms, partition and cause.

        ``raised_at`` is the frame that raised inside a user function, when one did.
        """
        self.op = op
        self.frames = tuple(frames)
        self.input_forms = tuple(input_forms)
        self.partition = partition
        self.cause_type = cause_type
        self.cause_message = cause_message
        self.opt_level = opt_level
        #: §8.1: the variation this failure belongs to. The EMPTY STRING is the single encoding of
        #: nominal/unvaried (never the literal ``"nominal"``); several universes sharing one
        #: failing node render sorted and joined by ``,`` (§8.2).
        self.variation = variation
        self.raised_at = raised_at
        super().__init__(self.summary())
        # set once, here: pickling carries it in __dict__ and a re-raise constructs nothing
        self.add_note(self.view())

    @property
    def user_frame(self) -> SourceFrame:
        """The top user frame — the exact analysis line that failed."""
        return self.frames[0] if self.frames else _NO_SOURCE_LINE

    def summary(self) -> str:
        """Return the one-line message naming the op, the user's line and the cause."""
        loc = self.user_frame
        variation = f", variation={self.variation}" if self.variation else ""
        return (
            f"StageError in op {self.op!r} at {loc} (partition {self.partition}{variation}, "
            f"opt_level={self.opt_level}): {self.cause_type}: {self.cause_message}"
        )

    def view(self) -> str:
        """Return the user's lines, failing one marked, then the cause, op, chunk and inputs."""
        lines: list[str] = []
        chain = list(reversed(self.frames))  # oldest ancestor first, failing line last
        for i, fr in enumerate(chain):
            marker = "  --> " if i == len(chain) - 1 else "      "
            lines.append(
                f'{marker}File "{fr.filename}", line {fr.lineno}, in {fr.function or "<module>"}'
            )
            if fr.source:
                lines.append(f"          {fr.source}")
            for other in fr.also:
                source = f":  {other.source}" if other.source else ""
                lines.append(f"      (same expression also written at {other}{source})")
        if not chain:
            lines.append(f"  --> {_NO_SOURCE_LINE.filename}")
        if (raised := self.raised_at) is not None:
            text = f":  {raised.source}" if raised.source else ""
            where = f"{raised.filename} line {raised.lineno}"
            lines.append(f"      raised in {raised.function}(), {where}{text}")
        lines.append(
            f"{self.cause_type}: {self.cause_message}  "
            f"[stage op {self.op!r}, partition {self.partition}, opt_level={self.opt_level}]"
        )
        if self.input_forms:
            lines.append(f"  input forms: {', '.join(self.input_forms)}")
        return "\n".join(lines)

    # Custom pickling: Exception's default uses self.args; we carry structured fields instead, so
    # the error round-trips byte-for-byte across a process boundary (only graphed.errors is needed
    # to load). `__new__` reconstructs a bare instance (no __init__), then __setstate__ restores
    # the fields.
    def __reduce__(self) -> tuple[object, tuple[type[StageError]], dict[str, object]]:
        """Pickle by state, so the error crosses a process boundary intact."""
        return (self.__class__.__new__, (self.__class__,), self.__dict__.copy())

    def __setstate__(self, state: dict[str, object] | None) -> None:
        """Restore the pickled state and the exception message."""
        if state:
            self.__dict__.update(state)
        Exception.__init__(self, self.summary())

    def __eq__(self, other: object) -> bool:
        """Compare two errors by every field."""
        if not isinstance(other, StageError):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """Hash the identifying fields."""
        return hash(
            (
                self.op,
                self.frames,
                self.partition,
                self.cause_type,
                self.cause_message,
                self.variation,
            )
        )


class WorkerLostError(StageError):
    """A worker died running a chunk (a crash, OOM kill or preemption).

    No user line failed, so ``frames`` is empty. ``partition`` is the chunk it held and ``worker``
    the backend's name for it. ``cause_type`` is ``"KilledWorker"`` on every backend.
    """

    def __init__(self, *, partition: str, worker: str) -> None:
        """Name the ``worker`` that died and the chunk it held."""
        self.worker = worker
        super().__init__(
            op="run",
            frames=(),
            input_forms=(),
            partition=partition,
            cause_type="KilledWorker",
            cause_message=(
                f"worker {worker} died (segfault, OOM kill or preemption suspected; blame can be "
                "unfair under co-located tasks)"
            ),
            opt_level=0,
        )

    def summary(self) -> str:
        """Return the one-line message naming the worker and the chunk."""
        return f"worker {self.worker} died running {self.partition}"


class PackKeyError(StageError, ValueError):
    """A :class:`PackKeyOverflowError` at its recording line, still a ``ValueError``."""
