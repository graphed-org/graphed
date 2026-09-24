"""graphed-checkpoint (plan M8): content-addressed checkpoint Store, deterministic resume, and
error harvesting on top of the M8 ``DurablePlan`` from ``graphed-core``.

``Store`` is a local directory; ``FsspecStore`` puts the same store at an fsspec URL, pulled forward
from Phase 2 past the M8 local-only guardrail. Analysis *preservation* is M9.
"""

from __future__ import annotations

from .codec import Codec, NumpyCodec, PickleCodec
from .errors import dead_letter_descriptor
from .fsspec_store import FsspecStore
from .retry import Quarantine, RetryElsewhere, RetryN, RetrySmallerChunk
from .runner import (
    ResumeReport,
    ResumeResult,
    ShuffleResumeResult,
    run_resumable,
    run_shuffle_resumable,
)
from .store import CheckpointStore, JournalEntry, Store

__all__ = [
    "CheckpointStore",
    "Codec",
    "FsspecStore",
    "JournalEntry",
    "NumpyCodec",
    "PickleCodec",
    "Quarantine",
    "ResumeReport",
    "ResumeResult",
    "RetryElsewhere",
    "RetryN",
    "RetrySmallerChunk",
    "ShuffleResumeResult",
    "Store",
    "dead_letter_descriptor",
    "run_resumable",
    "run_shuffle_resumable",
]
__version__ = "0.0.5"
