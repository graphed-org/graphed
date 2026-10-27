"""graphed frontend errors, and the worker-side ``StageError`` the frontend raises (plan M6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .provenance import Provenance


class GraphedError(Exception):
    """Base class for graphed frontend errors."""


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


class OutputTypeError(GraphedTypeError):
    """An External's value disagrees with the type its node declared.

    Located at the declaring line.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by constructor arguments."""
        return (type(self), (self.op, self.provenance, self.detail))


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

    def __str__(self) -> str:
        """Show the frame as ``filename:lineno``."""
        return f"{self.filename}:{self.lineno}"


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
    ) -> None:
        """Describe a failed op: its user frames, input forms, partition and cause."""
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
        super().__init__(self.summary())

    @property
    def user_frame(self) -> SourceFrame:
        """The top user frame — the exact analysis line that failed."""
        return self.frames[0]

    def summary(self) -> str:
        """Return the one-line message naming the op, the user's line and the cause."""
        loc = self.user_frame
        variation = f", variation={self.variation}" if self.variation else ""
        return (
            f"StageError in op {self.op!r} at {loc} (partition {self.partition}{variation}, "
            f"opt_level={self.opt_level}): {self.cause_type}: {self.cause_message}"
        )

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
