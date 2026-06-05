"""M8 — per-task partials are stored via a deterministic codec."""

from __future__ import annotations

import analyses
import numpy as np

from graphed_checkpoint import NumpyCodec, PickleCodec, Store, run_resumable


def test_pickle_codec_roundtrips_and_is_deterministic() -> None:
    c = PickleCodec()
    value = {"hist": [1, 2, 3], "n": 7}
    assert c.decode(c.encode(value)) == value
    assert c.encode(value) == c.encode(value)  # stable bytes -> stable content hash


def test_numpy_codec_roundtrips_and_is_deterministic() -> None:
    c = NumpyCodec()
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    out = c.decode(c.encode(arr))
    assert np.array_equal(out, arr)
    assert c.encode(arr) == c.encode(arr)


def test_runner_uses_a_supplied_numpy_codec(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = analyses.build_plan("analyses:histogram_chunk", 6)
    res = run_resumable(plan, Store(tmp_path), codec=NumpyCodec())
    assert np.array_equal(res.value, analyses.reference())
    # resume with the same codec reuses the .npy blobs (skips all work)
    again = run_resumable(plan, Store(tmp_path), codec=NumpyCodec())
    assert again.report.skipped == 6 and again.report.executed == 0
