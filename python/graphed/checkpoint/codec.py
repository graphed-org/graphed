"""Deterministic value <-> bytes codecs for per-task outputs in the Store (plan M8).

A per-task output (e.g. a histogram array) must serialize **deterministically** so the same result
always lands under the same content hash (so resume can recognize completed work and so an
uninterrupted run and a resumed run agree bit-for-bit). ``NumpyCodec`` uses ``numpy.save`` (a
fixed, versioned ``.npy`` layout); ``PickleCodec`` is the general fallback (protocol pinned).
"""

from __future__ import annotations

import io
import pickle
from typing import Any, Protocol


class Codec(Protocol):
    """Encode/decode a per-task partial result to/from bytes."""

    def encode(self, value: Any) -> bytes:
        """Encode ``value`` to bytes."""
        ...

    def decode(self, data: bytes) -> Any:
        """Decode bytes :meth:`encode` wrote."""
        ...


class PickleCodec:
    """General codec; protocol pinned so the byte layout is stable across runs."""

    PROTOCOL = 5

    def encode(self, value: Any) -> bytes:
        """Pickle ``value``."""
        return pickle.dumps(value, protocol=self.PROTOCOL)

    def decode(self, data: bytes) -> Any:
        """Unpickle ``data``."""
        return pickle.loads(data)


class NumpyCodec:
    """Deterministic codec for numpy arrays (the common histogram-partial case)."""

    def encode(self, value: Any) -> bytes:
        """Save ``value`` as ``.npy`` bytes, without pickling."""
        import numpy as np  # noqa: PLC0415

        buf = io.BytesIO()
        np.save(buf, np.asarray(value), allow_pickle=False)
        return buf.getvalue()

    def decode(self, data: bytes) -> Any:
        """Load ``.npy`` bytes, without unpickling."""
        import numpy as np  # noqa: PLC0415

        return np.load(io.BytesIO(data), allow_pickle=False)
