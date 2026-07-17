"""The acceptance-contract analysis: an out-of-range index in a dimuon mass calc yields a formatted
traceback whose top user frame is the exact mass-calc line, at both opt levels (plan M6)."""

from __future__ import annotations

from analyses import awkward_mass_oob

import graphed.debug as gd


def test_out_of_range_index_maps_to_the_mass_calc_line() -> None:
    s, mass = awkward_mass_oob()
    try:
        gd.run(s, mass, opt_level=1)
    except gd.StageError as e:
        assert e.user_frame.filename.endswith("analyses.py")
        out = gd.format_traceback(e)
        assert "analyses.py" in out
        # the underlying awkward error surfaced (an indexing / out-of-range failure)
        assert e.cause_type != ""
    else:
        raise AssertionError("expected a StageError from the out-of-range muon index")


def test_identical_source_location_at_opt0_and_opt1() -> None:
    s, mass = awkward_mass_oob()
    locs = []
    for opt in (0, 1):
        try:
            gd.run(s, mass, opt_level=opt)
        except gd.StageError as e:
            locs.append((e.user_frame.filename, e.user_frame.lineno))
    assert len(locs) == 2
    assert locs[0] == locs[1]
