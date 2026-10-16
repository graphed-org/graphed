"""torch and xgboost each vendor an OpenMP runtime; allow both to load in one process (as m26 does)."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
