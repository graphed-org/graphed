"""The debug runner maps a failure to the user's analysis line — including deep inside a fused stage,
and at the same location regardless of opt level (plan M6)."""

from __future__ import annotations

import numpy as np
from analyses import numpy_mismatch_in_fused_stage, numpy_oob

import graphed.debug as gd
import graphed.numpy as gn
from graphed import Session


def test_success_path_returns_the_value() -> None:
    s = Session(gn.NumpyBackend())
    ev = gn.from_record(s, "events", pt=np.array([1.0, 2.0, 3.0]))
    out = (ev["pt"] * 2.0).reduce("sum")
    assert gd.run(s, out, opt_level=1) == 12.0


def test_error_maps_to_the_user_analysis_line() -> None:
    s, bad = numpy_oob()
    try:
        gd.run(s, bad, opt_level=1)
    except gd.StageError as e:
        assert e.user_frame.filename.endswith("analyses.py")
        assert "oob_index" in e.user_frame.source or "map" in e.user_frame.source
        assert e.cause_type == "IndexError"
    else:
        raise AssertionError("expected a StageError")


def test_error_localizes_deep_inside_a_fused_stage() -> None:
    s, bad = numpy_mismatch_in_fused_stage()
    g = gd.lower(s, bad, opt_level=1)
    # the failing op is a non-boundary member fused into a multi-member stage
    failing_stage = next(st for st in g.stages if any(m.node_id == bad.node_id for m in st.members))
    assert len(failing_stage.members) >= 2
    try:
        gd.run(s, bad, opt_level=1)
    except gd.StageError as e:
        assert e.op == "add"  # the specific fused member, not the stage head or first op
        assert e.user_frame.lineno == g.op_for(bad.node_id).provenance.lineno
        assert e.cause_type == "ValueError"
    else:
        raise AssertionError("expected a StageError")


def test_same_source_location_at_opt0_and_opt1() -> None:
    s, bad = numpy_oob()
    locs = []
    for opt in (0, 1):
        try:
            gd.run(s, bad, opt_level=opt)
        except gd.StageError as e:
            locs.append((e.user_frame.filename, e.user_frame.lineno))
    assert locs[0] == locs[1]  # identical user source location, 1:1 vs fused


def test_opt0_reports_one_to_one_mapping() -> None:
    s, bad = numpy_oob()
    try:
        gd.run(s, bad, opt_level=0)
    except gd.StageError as e:
        assert e.opt_level == 0


def test_source_load_failure_is_mapped_to_a_stage_error() -> None:
    # a failing (lazy) source loader surfaces as a StageError pointing at the source, not a raw OSError
    s = Session(gn.NumpyBackend())

    def boom() -> object:
        raise OSError("cannot read dataset file")

    bad = s.source("dataset", form=gn.NumpyForm(np.dtype("float64")), data=boom)
    try:
        gd.run(s, bad, opt_level=1)
    except gd.StageError as e:
        assert e.cause_type == "OSError"
        assert e.op == "dataset"  # the source name
    else:
        raise AssertionError("expected a StageError from the failing source loader")
