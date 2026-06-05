"""Opt-level lowering (plan M6): opt_level=0 is 1:1; >=1 fuses; provenance preserved either way."""

from __future__ import annotations

import pytest
from analyses import numpy_mismatch_in_fused_stage, numpy_oob

import graphed_debug as gd


def test_opt_level_0_is_one_to_one() -> None:
    s, bad = numpy_oob()
    g = gd.lower(s, bad, opt_level=0)
    assert g.one_to_one
    assert all(len(stage.members) == 1 for stage in g.stages)
    # one stage per recorded op (source, field, mul, external)
    assert len(g.stages) == len(g.ops)


def test_opt_level_1_fuses_runs_between_boundaries() -> None:
    s, bad = numpy_oob()
    g0 = gd.lower(s, bad, opt_level=0)
    g1 = gd.lower(s, bad, opt_level=1)
    assert not g1.one_to_one
    assert len(g1.stages) < len(g0.stages)  # fusion reduces the node count
    # the field+mul chain fuses into one stage; the source and the external boundary stay separate
    assert any(len(stage.members) >= 2 for stage in g1.stages)


def test_provenance_is_identical_across_opt_levels() -> None:
    s, bad = numpy_oob()
    g0 = gd.lower(s, bad, opt_level=0)
    g1 = gd.lower(s, bad, opt_level=1)
    # the failing op keeps the SAME source location whether or not it is fused
    head = gd.lower(s, bad, opt_level=0).op_for(bad.node_id)
    assert g0.op_for(bad.node_id).provenance == head.provenance
    assert g1.op_for(bad.node_id).provenance == head.provenance


def test_fusion_never_crosses_a_boundary() -> None:
    s, bad = numpy_mismatch_in_fused_stage()
    g = gd.lower(s, bad, opt_level=1)
    for stage in g.stages:
        # at most the head of a stage may be a boundary; interior members are non-boundary ops
        assert all(not m.boundary for m in stage.members[:-1])


def test_negative_opt_level_rejected() -> None:
    s, bad = numpy_oob()
    with pytest.raises(ValueError):
        gd.lower(s, bad, opt_level=-1)


def test_op_for_unknown_node_raises() -> None:
    s, bad = numpy_oob()
    g = gd.lower(s, bad, opt_level=1)
    with pytest.raises(KeyError):
        g.op_for(999999)
