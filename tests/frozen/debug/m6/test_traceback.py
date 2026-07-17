"""`format_traceback` renders the user frames (collapsing graphed* internals), failing line last, and
is identical before and after a pickle round-trip (plan M6)."""

from __future__ import annotations

import pickle

from analyses import numpy_oob

from graphed.debug import SourceFrame, StageError, format_traceback, run


def _err() -> StageError:
    return StageError(
        op="index",
        frames=(
            SourceFrame("analysis.py", 42, "dimuon_mass", "mu[:, 1].pt"),
            SourceFrame("analysis.py", 40, "dimuon_mass", "events.Muon"),
        ),
        input_forms=("var * float64",),
        partition="file.root:Events[0:1000]",
        cause_type="IndexError",
        cause_message="index 1 is out of range",
        opt_level=0,
    )


def test_shows_the_failing_user_line_marked_and_last() -> None:
    out = format_traceback(_err())
    assert 'File "analysis.py", line 42, in dimuon_mass' in out
    assert "mu[:, 1].pt" in out
    # the failing line is marked and appears after the ancestor frame (most-recent-call-last)
    assert out.index("line 40") < out.index("line 42")
    assert (
        "-->"
        in out.splitlines()[out.splitlines().index(next(ln for ln in out.splitlines() if "line 42" in ln))]
    )


def test_includes_cause_and_partition() -> None:
    out = format_traceback(_err())
    assert "IndexError: index 1 is out of range" in out
    assert "file.root:Events[0:1000]" in out


def test_collapses_graphed_internal_frames() -> None:
    # the formatted user-frame trace contains only the user's file, never graphed* internal frames
    s, bad = numpy_oob()
    try:
        run(s, bad, opt_level=1)
    except StageError as e:
        out = format_traceback(e)
        # graphed* internal frames are collapsed (provenance.capture skips module.startswith("graphed")).
        # The consolidated repo dir is itself named "graphed", so the user file's own path contains
        # "graphed/"; assert no INTERNAL package SOURCE file leaks instead of a bare "graphed/" substring.
        internal_sources = (
            "session.py",
            "array.py",
            "aggregate.py",
            "runner.py",
            "lowering.py",
            "execute.py",
        )
        assert not any(src in out for src in internal_sources)
        assert "site-packages" not in out
        assert "analyses.py" in out


def test_identical_after_pickle_round_trip() -> None:
    e = _err()
    assert format_traceback(e) == format_traceback(pickle.loads(pickle.dumps(e)))
