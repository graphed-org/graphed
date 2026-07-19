"""M40 determinism + backend-independence gate for ``graphed.join`` (contract determinism row).

Two witnesses, both over the REAL backends:

* across-runs determinism — two independent builds of the same join compile to a byte-identical
  reduced IR (the M8 durable-artifact guarantee; a nondeterministic router / task-id fold drifts).
* backend-independence — the recorded join graph is structurally identical across AwkwardBackend and
  NumpyBackend (the M2 ``to_dot`` witness): the graph structure is backend-independent, only forms /
  evaluation differ. This is the ShuffleBackend-seam witness at the recording layer.

Pre-implementation ``graphed`` has no ``join`` attribute -> right-reason ``AttributeError``.
"""

from __future__ import annotations

import pytest
from shuffle_backends import ON, REAL_BACKENDS, BackendCase, skim_tables

import graphed
from graphed import Session, compile_ir

_IDS = [c.name for c in REAL_BACKENDS]


def _build(case: BackendCase) -> tuple[Session, object]:
    left_cols, right_cols = skim_tables()
    s = Session(case.make_backend())
    left = case.make_source(s, "left", left_cols)
    right = case.make_source(s, "right", right_cols)
    return s, graphed.join(left, right, on=ON, how="inner")


@pytest.mark.parametrize("case", REAL_BACKENDS, ids=_IDS)
def test_recorded_join_ir_is_byte_identical_across_runs(case: BackendCase) -> None:
    s1, j1 = _build(case)
    s2, j2 = _build(case)
    ir1 = bytes(compile_ir(s1, j1).ir)
    ir2 = bytes(compile_ir(s2, j2).ir)
    assert ir1 == ir2, f"{case.name}: identical input must compile to a byte-identical join IR"


def test_recorded_join_graph_is_backend_independent() -> None:
    # same program, two backends -> byte-identical recorded structure (M2 discipline). A join that
    # leaked backend-specific structure into the recorded graph would differ here.
    dots = [_build(c)[0].to_dot() for c in REAL_BACKENDS]
    assert dots[0] == dots[1], "the recorded join graph must be identical across awkward and numpy"
