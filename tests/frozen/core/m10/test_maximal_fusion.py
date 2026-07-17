"""M10 — opt-in maximal stage fusion (finding C.6).

The M4 single-use rule splits a stage at every fan-out, so a diamond inside an op region becomes
three stages where the glossary's "maximal fused run" allows one. `reduce(maximal_fusion=True)`
fuses a fan-out op when ALL of its consumers are ops that land in one stage. The default stays the
single-use rule — the frozen M4 suite pins it, and this suite re-asserts the default so the opt-in
can never silently become a behavior change.
"""

from __future__ import annotations

from graphed_core import GraphStore


def _diamond() -> tuple[GraphStore, int]:
    s = GraphStore()
    x = s.add_source("x")
    a = s.add_op("inc", [x])  # fan-out apex
    left = s.add_op("inc", [a])
    right = s.add_op("neg", [a])
    out = s.add_op("add", [left, right])
    return s, out


def test_diamond_fuses_to_one_stage_with_maximal_fusion() -> None:
    s, out = _diamond()
    reduced, report = s.reduce(maximal_fusion=True, outputs=[out])
    assert report["stages"] == 1, "the whole diamond is one maximal op-run"
    assert reduced.node_count() == 2  # source + one stage
    stage = next(n for n in reduced.nodes() if n["kind"] == "stage")
    # the apex is a member referenced by BOTH branches — fused, not duplicated
    assert stage["n_members"] == 4
    apex_refs = [ref for m in stage["members"] for ref in m["inputs"] if ref == ("member", 0)]
    assert len(apex_refs) == 2


def test_default_mode_is_unchanged_single_use() -> None:
    # the M4 frozen pin: apex stays its own stage by default
    s, out = _diamond()
    _, report = s.reduce(outputs=[out])
    assert report["stages"] == 2
    s1, o1 = _diamond()
    s2, o2 = _diamond()
    assert s1.reduce(outputs=[o1])[0].serialize() == s2.reduce(outputs=[o2])[0].serialize()


def test_maximal_fusion_never_crosses_a_boundary() -> None:
    s = GraphStore()
    x = s.add_source("x")
    pre = s.add_op("inc", [x])
    red = s.add_reduction("sum", [pre])  # boundary
    post = s.add_op("inc", [red])
    _, report = s.reduce(maximal_fusion=True, outputs=[post])
    assert report["stages"] == 2, "a reduction still splits stages"


def test_fan_out_to_a_boundary_still_splits() -> None:
    # apex feeds an op AND a reduction -> not all consumers are ops -> apex heads its own stage
    # (while the pure op chain b -> out still fuses): stage(a) + stage(b, out)
    s = GraphStore()
    x = s.add_source("x")
    a = s.add_op("inc", [x])
    b = s.add_op("neg", [a])
    r = s.add_reduction("sum", [a])
    out = s.add_op("add", [b, r])
    _, report = s.reduce(maximal_fusion=True, outputs=[out])
    assert report["stages"] == 2
    assert report["boundary_nodes"] == 2  # source + reduction


def test_fan_out_to_two_different_stages_does_not_fuse() -> None:
    # apex feeds ops in TWO different stages (split by a boundary downstream of one branch)
    s = GraphStore()
    x = s.add_source("x")
    a = s.add_op("inc", [x])
    left = s.add_op("inc", [a])
    red = s.add_reduction("sum", [left])
    right = s.add_op("neg", [a])
    out = s.add_op("add", [red, right])
    _, report = s.reduce(maximal_fusion=True, outputs=[out])
    # left fuses upstream of the boundary; right/out fuse after it; the apex's consumers live in
    # different stages, so it stays its own stage: 3 stages + 1 boundary
    assert report["stages"] == 3
    assert report["boundary_nodes"] == 2  # source + reduction


def test_maximal_fusion_is_deterministic_and_durable() -> None:
    def run() -> bytes:
        s, out = _diamond()
        return bytes(s.reduce(maximal_fusion=True, outputs=[out])[0].serialize())

    blob = run()
    assert blob == run()
    assert GraphStore.deserialize(blob).serialize() == blob
