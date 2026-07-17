"""M39 (e) — resume a half-done shuffle = bit-for-bit vs uninterrupted (plan §5.3, §7.3).

The M8 kill/resume pattern extended to two phases: a resumed run reuses every already-journaled
map-write AND gather block (``skipped > 0`` witnesses reuse), runs only the unfinished ones, and
produces the byte-identical result — because every block is content-addressed, so a crash at any
point resumes from the last durable block.

Pinned surface: ``run_shuffle_resumable(plan_v2, store, *, resources=None, _kill_after=None) ->
ShuffleResumeResult`` with ``.value`` (the tuple of content-addressed gather-block hashes) and
``.report`` (an M8-style report exposing ``executed``/``skipped``).
"""

from __future__ import annotations

import pytest
import shuffle_analyses as sa

from graphed.checkpoint import Store, run_shuffle_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt

N_PROD, N_DEST = 3, 2
_TOTAL = sa.n_blocks(N_PROD, N_DEST)  # 3 map-write + 2 gather = 5 blocks


def _plan():  # type: ignore[no-untyped-def]
    return sa.build_shuffle_plan_v2(N_PROD, N_DEST)


def _reference_value(tmp_path):  # type: ignore[no-untyped-def]
    return run_shuffle_resumable(_plan(), Store(tmp_path / "ref")).value


def test_uninterrupted_run_executes_every_block(tmp_path) -> None:  # type: ignore[no-untyped-def]
    res = run_shuffle_resumable(_plan(), Store(tmp_path))
    assert res.report.executed == _TOTAL and res.report.skipped == 0
    assert len(res.value) == N_DEST, "the shuffle yields one gather block per dest partition"


def test_kill_then_resume_reuses_blocks_and_is_bit_for_bit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = Store(tmp_path / "run")
    ref = _reference_value(tmp_path)
    # crash after 3 blocks commit (the 3 map-write blocks, i.e. mid-shuffle before the gather).
    with pytest.raises(_SimulatedInterrupt):
        run_shuffle_resumable(_plan(), store, _kill_after=3)
    assert len(store.completed()) == 3, "only committed blocks survived the kill"

    res = run_shuffle_resumable(_plan(), store)  # resume on the same store
    assert res.report.skipped == 3, "the journaled map-write blocks must be REUSED, not recomputed"
    assert res.report.executed == _TOTAL - 3, "only the unfinished (gather) blocks run"
    assert res.report.did_less_work
    assert res.value == ref, "the resumed shuffle result is byte-identical to the uninterrupted run"


def test_resume_after_completion_redoes_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = Store(tmp_path)
    first = run_shuffle_resumable(_plan(), store)
    again = run_shuffle_resumable(_plan(), store)
    assert again.report.executed == 0 and again.report.skipped == _TOTAL
    assert again.value == first.value


def test_kill_at_every_boundary_still_matches_uninterrupted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ref = _reference_value(tmp_path)
    for k in range(1, _TOTAL):
        store = Store(tmp_path / f"k{k}")
        with pytest.raises(_SimulatedInterrupt):
            run_shuffle_resumable(_plan(), store, _kill_after=k)
        res = run_shuffle_resumable(_plan(), store)
        assert res.report.skipped == k, f"kill after {k} must reuse exactly {k} blocks"
        assert res.value == ref
