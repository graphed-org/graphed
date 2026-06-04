"""graphed: deferred-array frontend that records a backend-agnostic program into graphed-core.

No fusion (that is M4); no awkward (the awkward backend is M3); provenance is a stub (M3).
"""

from __future__ import annotations

from .array import Array
from .backend import Backend, Form
from .errors import GraphedError, GraphedTypeError
from .provenance import Provenance, capture
from .session import Session

__all__ = [
    "Array",
    "Backend",
    "Form",
    "GraphedError",
    "GraphedTypeError",
    "Provenance",
    "Session",
    "capture",
]

__version__ = "0.0.1"
