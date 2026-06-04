"""graphed-corpus: ground-truth requirements + runnable canonical-analysis fixtures (M0.5).

Distilled from the A.8 reference corpus. Provides a deterministic synthetic dataset, the ADL
queries 1-8, and AGC-ttbar-style / TTGamma-style systematics slices, each with a stored reference
histogram so later milestones can assert ``graphed`` reproduces plain awkward bit-for-bit.
"""

from __future__ import annotations

from .analyses import ADL_QUERIES, TTBAR_FIXTURES, TTGAMMA_FIXTURES, ttbar_region, ttgamma_region
from .dataset import make_events
from .histograms import fingerprint, hist1d, load_reference, reference_record, write_reference

__all__ = [
    "ADL_QUERIES",
    "TTBAR_FIXTURES",
    "TTGAMMA_FIXTURES",
    "fingerprint",
    "hist1d",
    "load_reference",
    "make_events",
    "reference_record",
    "ttbar_region",
    "ttgamma_region",
    "write_reference",
]

__version__ = "0.0.1"
