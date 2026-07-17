"""Lowering + error localization through complex topologies (plan M6). A diamond's fan-out apex must
be a single shared stage head (never duplicated), and an error in ONE branch of a diamond must
localize to that branch's op — the other branch must not mask or mislocate it."""

from __future__ import annotations

import graphed_numpy as gn
import numpy as np
from graphed import Session

import graphed_debug as gd


def _events(s: Session) -> object:
    return gn.from_record(s, "events", pt=np.arange(1.0, 5.0), eta=np.linspace(0, 1, 4))


def test_diamond_apex_is_a_single_shared_stage_head() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    apex = ev["pt"] * 2.0  # fans out to two branches
    out = (apex + 1.0) + (apex - 1.0)
    g = gd.lower(s, out, opt_level=1)
    apex_op = g.op_for(apex.node_id)
    # the apex appears exactly once across all stages (not duplicated into each branch)
    occurrences = sum(1 for op in g.ops if op.node_id == apex.node_id)
    assert occurrences == 1
    # and it is the head of its own stage (out-degree 2 ends a fused run)
    apex_stage = next(st for st in g.stages if any(m.node_id == apex.node_id for m in st.members))
    assert apex_stage.head.node_id == apex_op.node_id


def test_star_hub_is_shared_in_lowering() -> None:
    s = Session(gn.NumpyBackend())
    ev = _events(s)
    hub = ev["pt"] * 2.0
    out = hub
    for _ in range(8):
        out = out + (hub + ev["eta"])
    g = gd.lower(s, out, opt_level=1)
    assert sum(1 for op in g.ops if op.node_id == hub.node_id) == 1  # one shared hub node


def _diamond_with_bad_branch(s: Session) -> object:
    # the bad branch is a shape-mismatch add (numeric form at record time, raises at eval), so both
    # branches re-converge numerically; apex fans out to the good and the bad branch.
    ev = _events(s)
    other = gn.from_record(s, "other", q=np.arange(7.0))  # length 7 vs events' length 4
    apex = ev["pt"] * 2.0
    good = apex + 1.0  # the healthy branch
    bad = apex + other["q"]  # <-- the failing branch (len-4 + len-7 mismatch at eval)
    return good + bad


def test_error_in_one_diamond_branch_localizes_to_that_branch() -> None:
    s = Session(gn.NumpyBackend())
    out = _diamond_with_bad_branch(s)
    try:
        gd.run(s, out, opt_level=1)
    except gd.StageError as e:
        assert e.cause_type == "ValueError"  # the numpy broadcast error from the bad branch
        assert e.op == "add"
        assert e.user_frame.filename.endswith("test_topologies.py")
    else:
        raise AssertionError("expected a StageError from the failing diamond branch")


def test_error_localization_is_opt_level_invariant_in_a_diamond() -> None:
    locs = []
    for opt in (0, 1):
        s = Session(gn.NumpyBackend())
        out = _diamond_with_bad_branch(s)
        try:
            gd.run(s, out, opt_level=opt)
        except gd.StageError as e:
            locs.append((e.user_frame.filename, e.user_frame.lineno))
    assert len(locs) == 2 and locs[0] == locs[1]
