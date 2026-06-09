"""Necessary-buffer (column) projection support (plan M5).

The frontend supplies a backend-agnostic graph `walk` and the `Projection` result type; each backend
computes the actually-touched columns its own way (graphed-awkward via a reporting typetracer,
graphed-numpy via record-field touch tracking). Opaque ops (`map`) cannot be projected through, so a
configurable on-fail policy (`pass | warn | raise`) governs them — mirroring dask-awkward.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .errors import GraphedError


class OnFail(StrEnum):
    PASS = "pass"
    WARN = "warn"
    RAISE = "raise"


class ProjectionError(GraphedError):
    """Raised when projection cannot proceed (e.g. an opaque op under the ``raise`` policy)."""


@dataclass(frozen=True)
class Projection:
    """The columns each source must actually read. ``read_columns`` maps source name -> column set."""

    read_columns: Mapping[str, frozenset[str]]

    def columns_for(self, source: str) -> frozenset[str]:
        return frozenset(self.read_columns.get(source, frozenset()))

    def total_columns(self) -> int:
        return sum(len(cols) for cols in self.read_columns.values())


class BufferNeed(StrEnum):
    """How much of a column a graph actually needs (M10, buffer-level projection).

    ``DATA``: the leaf values (implies the structure needed to interpret them).
    ``OFFSETS``: only the list structure — e.g. a multiplicity/count; the leaf values are never
    read. At column granularity this case collapsed to "nothing", which under-specifies the read
    (a count-only analysis projected to the empty column set)."""

    DATA = "data"
    OFFSETS = "offsets"


@dataclass(frozen=True)
class BufferProjection:
    """Buffer-granular projection (M10): per source, each needed column with its `BufferNeed`.

    Strictly finer than :class:`Projection`: `columns_for` (data-bearing columns) reproduces the
    column-level view, while `offsets_only_for` names the columns whose *structure alone* is needed
    — readable as a counter branch (TTree) or an index column (RNTuple) without the payload."""

    read_buffers: Mapping[str, Mapping[str, BufferNeed]]

    def buffers_for(self, source: str) -> dict[str, BufferNeed]:
        return dict(self.read_buffers.get(source, {}))

    def columns_for(self, source: str) -> frozenset[str]:
        """Columns whose leaf DATA is read (the column-level view)."""
        return frozenset(
            c for c, need in self.read_buffers.get(source, {}).items() if need is BufferNeed.DATA
        )

    def offsets_only_for(self, source: str) -> frozenset[str]:
        """Columns needed only for their list structure (counts), never their values."""
        return frozenset(
            c for c, need in self.read_buffers.get(source, {}).items() if need is BufferNeed.OFFSETS
        )

    def to_projection(self) -> Projection:
        """Collapse to the column-level :class:`Projection` (data-bearing columns only)."""
        return Projection({src: frozenset(self.columns_for(src)) for src in self.read_buffers})


# sentinel returned by `handle_opaque` to signal "conservatively assume everything is read"
CONSERVATIVE = object()


def handle_opaque(op: str, on_fail: str, detail: str = "") -> object | None:
    """Apply the on-fail policy when projection hits an opaque op. Returns CONSERVATIVE for ``warn``,
    None for ``pass``, and raises for ``raise``."""
    policy = OnFail(on_fail)
    if policy is OnFail.RAISE:
        msg = f"cannot project through opaque op {op!r}"
        raise ProjectionError(f"{msg}: {detail}" if detail else msg)
    if policy is OnFail.WARN:
        warnings.warn(
            f"projecting through opaque op {op!r}; conservatively reading all columns",
            stacklevel=3,
        )
        return CONSERVATIVE
    return None
