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
