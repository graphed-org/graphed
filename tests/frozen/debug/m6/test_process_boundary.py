"""The headline guarantee (plan A.3 #8 / M6): a `StageError` raised in a separate worker PROCESS is
re-raised in the driver as the SAME formatted traceback pointing at the user's analysis line — never
a raw, opaque worker traceback. This exercises a real pickle-across-a-process-boundary round-trip."""

from __future__ import annotations

import multiprocessing as mp

import analyses

import graphed_debug as gd


def test_stage_error_survives_a_real_worker_process() -> None:
    ctx = mp.get_context("spawn")  # spawn: a genuinely separate interpreter, cross-platform
    with ctx.Pool(processes=1) as pool:
        result = pool.apply_async(analyses.run_numpy_oob_and_raise)
        try:
            result.get(timeout=120)
        except gd.StageError as e:
            # the error crossed the process boundary intact (structured, not an opaque string)
            assert isinstance(e, gd.StageError)
            assert e.cause_type == "IndexError"
            assert e.user_frame.filename.endswith("analyses.py")
            out = gd.format_traceback(e)
            assert "analyses.py" in out
            assert "IndexError" in out
            # the driver renders the USER source, not a worker/multiprocessing traceback
            assert "multiprocessing" not in out and "site-packages" not in out
        else:
            raise AssertionError("expected a StageError re-raised from the worker process")
