"""M26 suite environment: torch/xgboost/tensorflow each vendor an OpenMP runtime; set the
coexistence guards BEFORE any of them load (the m9 precedent), harmless when absent."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
