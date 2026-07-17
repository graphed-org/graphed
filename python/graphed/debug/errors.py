"""``StageError`` — the picklable, source-mapped error that survives a process boundary (plan M6).

A runtime failure inside a stage is wrapped in a ``StageError`` carrying the failing op, the **user
source-frame chain** (filename / line / function / sub-expression text), the input forms, and the
partition — all as plain data, so the error pickles intact and can be re-raised in the driver as the
same formatted traceback pointing at the user's analysis line. It never degrades to an opaque string
(plan A.3 #8). It deliberately does NOT carry the live Python traceback object (those do not pickle);
the source-frame chain is the durable, transportable record.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFrame:
    """One user-source frame: where a node was created in the user's analysis."""

    filename: str
    lineno: int
    function: str = ""
    source: str = ""

    def __str__(self) -> str:
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
    ) -> None:
        self.op = op
        self.frames = tuple(frames)
        self.input_forms = tuple(input_forms)
        self.partition = partition
        self.cause_type = cause_type
        self.cause_message = cause_message
        self.opt_level = opt_level
        super().__init__(self.summary())

    @property
    def user_frame(self) -> SourceFrame:
        """The top user frame — the exact analysis line that failed."""
        return self.frames[0]

    def summary(self) -> str:
        loc = self.user_frame
        return (
            f"StageError in op {self.op!r} at {loc} (partition {self.partition}, "
            f"opt_level={self.opt_level}): {self.cause_type}: {self.cause_message}"
        )

    # Custom pickling: Exception's default uses self.args; we carry structured fields instead, so the
    # error round-trips byte-for-byte across a process boundary (only graphed.debug is needed to load).
    # `__new__` reconstructs a bare instance (no __init__), then __setstate__ restores the fields.
    def __reduce__(self) -> tuple[object, tuple[type[StageError]], dict[str, object]]:
        return (self.__class__.__new__, (self.__class__,), self.__dict__.copy())

    def __setstate__(self, state: dict[str, object] | None) -> None:
        if state:
            self.__dict__.update(state)
        Exception.__init__(self, self.summary())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StageError):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((self.op, self.frames, self.partition, self.cause_type, self.cause_message))
