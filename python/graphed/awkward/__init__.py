"""graphed-awkward: the reference backend (awkward typetracer forms + real evaluation), plan M3.

``op_form`` uses the awkward **typetracer** (metadata only -- no event data is read); ``eval_stage``
uses real awkward. The ``gak`` namespace mirrors the awkward API so corpus analyses record a
backend-agnostic graph. External inputs (correctionlib corrections, ONNX models) record ``External``
nodes with a content-hashed ``PayloadDescriptor``. Reuse awkward/correctionlib/ONNX -- invent nothing.
"""

from __future__ import annotations

from . import functions, gnano, io, payloads, shuffle
from . import functions as gak
from .backend import AwkwardBackend, AwkwardForm, from_awkward
from .io import from_parquet, read_parquet_partition, read_varied, to_parquet
from .projection import project, project_buffers

#: §2.3d: the awkward idiom's public `Array`-consuming module verbs, each answering PER LABEL
#: with its OWN return type (`Projection` / `BufferProjection`).
VERB_DISPOSITIONS: dict[str, str] = {"project": "expanding", "project_buffers": "expanding"}

__all__ = [
    "VERB_DISPOSITIONS",
    "AwkwardBackend",
    "AwkwardForm",
    "from_awkward",
    "from_parquet",
    "functions",
    "gak",
    "gnano",
    "io",
    "payloads",
    "project",
    "project_buffers",
    "read_parquet_partition",
    "read_varied",
    "shuffle",
    "to_parquet",
]

__version__ = "0.0.1"
