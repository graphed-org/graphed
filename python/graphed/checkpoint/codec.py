"""Deterministic value <-> bytes codecs for per-task outputs in the Store (plan M8).

A per-task output (e.g. a histogram array) must serialize **deterministically** so the same result
always lands under the same content hash (so resume can recognize completed work and so an
uninterrupted run and a resumed run agree bit-for-bit). ``NumpyCodec`` uses ``numpy.save`` (a
fixed, versioned ``.npy`` layout); ``PickleCodec`` is the general fallback (protocol pinned).
"""

from __future__ import annotations

import io
import pickle
import sys
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
        """Pickle ``value``, each awkward array in it with its form's parameters in key order."""
        buf = io.BytesIO()
        _OrderedPickler(buf, protocol=self.PROTOCOL).dump(value)
        return buf.getvalue()

    def decode(self, data: bytes) -> Any:
        """Unpickle ``data``."""
        return pickle.loads(data)


class _OrderedPickler(pickle.Pickler):
    """The pickler whose awkward arrays carry their parameters in key order, not hash order."""

    def reducer_override(self, obj: Any) -> Any:
        ak = sys.modules.get("awkward")  # no awkward imported: no awkward array to reorder
        # behavior classes subclass ak.Array and ak.Record
        if ak is None or not isinstance(obj, (ak.Array, ak.Record)):
            return NotImplemented
        from ..awkward._form_order import ordered_array, ordered_record

        ordered = ordered_array(obj) if isinstance(obj, ak.Array) else ordered_record(obj)
        return ordered.__reduce_ex__(PickleCodec.PROTOCOL)


class NumpyCodec:
    """Deterministic codec for numpy arrays (the common histogram-partial case)."""

    def encode(self, value: Any) -> bytes:
        """Save ``value`` as ``.npy`` bytes, without pickling."""
        import numpy as np

        buf = io.BytesIO()
        np.save(buf, np.asarray(value), allow_pickle=False)
        return buf.getvalue()

    def decode(self, data: bytes) -> Any:
        """Load ``.npy`` bytes, without unpickling."""
        import numpy as np

        return np.load(io.BytesIO(data), allow_pickle=False)
