"""Provenance capture (STUB for M2; the rich version is M3).

M2 captures only the first non-`graphed*` stack frame's (filename, lineno) — enough to point a
build-time type error at the user's analysis line. M3 replaces this with thread-safe sub-expression
capture (stack_data/executing), toggling, and benchmarking.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    filename: str
    lineno: int

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}"


def capture() -> Provenance:
    """Return the first stack frame outside the graphed* packages (the user's line)."""
    for info in inspect.stack()[1:]:
        module = info.frame.f_globals.get("__name__", "")
        if not module.startswith("graphed"):
            return Provenance(info.filename, info.lineno)
    return Provenance("<unknown>", 0)
