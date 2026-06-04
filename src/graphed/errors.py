"""graphed frontend errors."""

from __future__ import annotations

from .provenance import Provenance


class GraphedError(Exception):
    """Base class for graphed frontend errors."""


class GraphedTypeError(GraphedError):
    """An ill-typed op, raised at the user's source line (plan M2)."""

    def __init__(self, op: str, provenance: Provenance, detail: str = "") -> None:
        self.op = op
        self.provenance = provenance
        self.detail = detail
        msg = f"ill-typed op {op!r} at {provenance}"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)
