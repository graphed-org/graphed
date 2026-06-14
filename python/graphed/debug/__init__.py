"""graphed-debug (plan M6): opt-level lowering, source-mapped picklable tracebacks, visualization.

The headline guarantee: a runtime error — even one raised deep inside a fused stage on a remote
worker process — is re-raised in the driver as a `StageError` whose `format_traceback` points at the
user's analysis line, never a raw opaque worker traceback (plan A.3 #8).
"""

from __future__ import annotations

from .dashboard import Dashboard, DashboardServer, NetworkMonitor
from .errors import SourceFrame, StageError
from .lowering import LoweredGraph, LoweredOp, LoweredStage, lower
from .runner import run
from .tracebacks import format_traceback
from .viz import visualize

__all__ = [
    "Dashboard",
    "DashboardServer",
    "LoweredGraph",
    "LoweredOp",
    "LoweredStage",
    "NetworkMonitor",
    "SourceFrame",
    "StageError",
    "format_traceback",
    "lower",
    "run",
    "visualize",
]
__version__ = "0.0.1"
