"""graphed frontend errors."""

from __future__ import annotations

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
        self.situation = situation
        self.entry = entry
        self.valid = valid
        self.detail = detail
        super().__init__(f"{situation}: {detail}" if detail else situation)


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
