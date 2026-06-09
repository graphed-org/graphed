"""graphed-awkward: the reference backend (awkward typetracer forms + real evaluation), plan M3.

`op_form` uses the awkward **typetracer** (metadata only — no event data is read); `eval_stage` uses
real awkward. The `gak` namespace mirrors the awkward API so corpus analyses record a
backend-agnostic graph. External inputs (correctionlib corrections, ONNX models) record `External`
nodes with content-hashed `PayloadDescriptor`s. Reuse awkward/correctionlib/ONNX — invent nothing.
"""

from __future__ import annotations

from . import functions, payloads
from . import functions as gak
from .backend import AwkwardBackend, AwkwardForm, from_awkward, from_parquet
from .projection import project, project_buffers

__all__ = [
    "AwkwardBackend",
    "AwkwardForm",
    "from_awkward",
    "from_parquet",
    "functions",
    "gak",
    "payloads",
    "project",
    "project_buffers",
]

__version__ = "0.0.1"
