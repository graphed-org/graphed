"""§8.2(ii)/(iii) across a REAL process boundary: the label and the user's analysis line survive the
pickle round-trip, so the M6 guarantee (plan A.3 #8) is EXTENDED by attribution, never narrowed.

The worker builds and runs the failing plan in a genuinely separate spawned interpreter and lets the
attributed ``StageError`` propagate; the parent rebuilds the same program only to read the expected
line out of the sourcemap.
"""

from __future__ import annotations

import multiprocessing as mp

import m49_analyses as m49

import graphed.debug as gd


def test_a_labelled_stage_error_survives_a_spawned_worker_process() -> None:
    session, _out, poisoned, _src = m49.merged_key_program()
    expected_line = session.sourcemap()[poisoned.node_id]["lineno"]

    ctx = mp.get_context("spawn")  # spawn: a genuinely separate interpreter, cross-platform
    with ctx.Pool(processes=1) as pool:
        result = pool.apply_async(m49.run_labelled_failure_and_raise)
        try:
            result.get(timeout=120)
        except gd.StageError as e:
            assert e.variation == m49.LABEL
            assert e.cause_type == "PoisonError"
            assert e.cause_message == m49.POISON_MESSAGE
            assert e.user_frame.filename.endswith("m49_analyses.py")
            assert e.user_frame.lineno == expected_line
            out = gd.format_traceback(e)
            assert "m49_analyses.py" in out
            assert "PoisonError" in out
            # the driver renders the USER source, not a worker/multiprocessing traceback
            assert "multiprocessing" not in out and "site-packages" not in out
        else:
            raise AssertionError("expected a labelled StageError re-raised from the worker process")
