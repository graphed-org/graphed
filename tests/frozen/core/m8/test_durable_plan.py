"""M8 — the durable, serializable, content-addressed Plan.

Pins the plan-level guarantees of plan M8:
- versioned deterministic serialization, **byte-identical for identical plans**;
- content-addressed ``task_id`` that is cache-poisoning-safe (changes iff computation/input changes,
  distinct computations never collide);
- the canonical durable form is the IR; cloudpickle is used **only** for genuinely-opaque callables,
  which are flagged ``opaque=True`` (a preservation risk, plan A.3.1);
- a serialized plan reconstructs and its callables resolve **on a process with no user source files**.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from graphed_core import DurablePlan, GraphStore, OpSpec, Partition


def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "data.root"})
    pt = g.add_op("pt", [src])
    out = g.add_reduction("sum", [pt])
    return g.serialize(outputs=[out])


def _plan(ir: bytes | None = None) -> DurablePlan:
    return DurablePlan(
        ir=ir if ir is not None else _ir(),
        process=OpSpec.from_ref("math:hypot"),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:float"),
        partitions=(Partition("data.root", "Events", 0, 100), Partition("data.root", "Events", 100, 200)),
        read_columns=("Jet_pt", "MET_pt"),
        stopping={"error_budget": 3},
        file_locality={"data.root": "node-a"},
        resource_hints={"memory_gb": 4},
    )


# ---- determinism / round trip -------------------------------------------------------------------
def test_serialization_is_byte_identical_for_identical_plans() -> None:
    assert _plan().to_bytes() == _plan().to_bytes()


def test_roundtrip_recovers_every_field() -> None:
    p = _plan()
    q = DurablePlan.from_bytes(p.to_bytes())
    assert q.to_bytes() == p.to_bytes()
    assert q.read_columns == p.read_columns
    assert tuple(q.partitions) == tuple(p.partitions)
    assert q.stopping == p.stopping
    assert q.file_locality == p.file_locality
    assert q.resource_hints == p.resource_hints
    assert q.format_version == p.format_version


def test_fingerprint_changes_with_metadata_but_is_stable() -> None:
    p = _plan()
    assert p.fingerprint() == _plan().fingerprint()
    other = DurablePlan(
        ir=p.ir,
        process=p.process,
        combine=p.combine,
        empty=p.empty,
        partitions=p.partitions,
        read_columns=("Jet_pt",),  # fewer columns -> different plan
    )
    assert other.fingerprint() != p.fingerprint()


def test_graph_reconstructs_from_the_plan_alone() -> None:
    p = _plan()
    g = p.graph()
    assert g.to_dot() == GraphStore.deserialize(p.ir).to_dot()


# ---- content-addressed task_id ------------------------------------------------------------------
def test_task_id_is_deterministic_and_per_partition() -> None:
    p = _plan()
    a, b = p.partitions[0], p.partitions[1]
    assert p.task_id(a) == p.task_id(a)
    assert p.task_id(a) != p.task_id(b), "different partitions -> different task ids"
    assert len(p.task_id(a)) == 64  # full sha-256 hex


def test_task_id_changes_with_the_computation() -> None:
    # same partition, different IR (a different analysis) -> different id (cache-poisoning-safe)
    part = Partition("data.root", "Events", 0, 100)
    g2 = GraphStore()
    s = g2.add_source("events", {"uri": "data.root"})
    out2 = g2.add_op("eta", [s])  # a *different* computation
    assert _plan().task_id(part) != _plan(g2.serialize(outputs=[out2])).task_id(part)


def test_task_id_changes_with_the_process_callable() -> None:
    part = Partition("data.root", "Events", 0, 100)
    base = _plan()
    other = DurablePlan(
        ir=base.ir,
        process=OpSpec.from_ref("math:atan2"),  # different process
        combine=base.combine,
        empty=base.empty,
        partitions=base.partitions,
    )
    assert base.task_id(part) != other.task_id(part)


# ---- IR-vs-cloudpickle policy -------------------------------------------------------------------
def test_importable_callable_is_referenced_not_pickled() -> None:
    spec = OpSpec.from_callable(math.hypot)
    assert spec.kind == "ref" and not spec.opaque
    assert spec.resolve() is math.hypot


def test_only_opaque_callables_are_embedded_by_value() -> None:
    base = 10

    def closure(x: int) -> int:  # not importable -> must be embedded by value
        return x + base

    spec = OpSpec.from_callable(closure)
    assert spec.kind == "opaque" and spec.opaque
    assert spec.resolve()(5) == 15


def test_plan_opaque_flag_reflects_any_opaque_callable() -> None:
    assert not _plan().opaque
    opaque_proc = DurablePlan(
        ir=_ir(),
        process=OpSpec(kind="opaque", blob_b64=OpSpec.from_callable(lambda x: x).blob_b64),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:float"),
    )
    assert opaque_proc.opaque


def test_opaque_process_changes_task_id_and_roundtrips() -> None:
    # exercises the opaque branch of OpSpec.identity + a full opaque round trip through bytes
    part = Partition("data.root", "Events", 0, 100)
    ref_plan = _plan()
    opaque_plan = DurablePlan(
        ir=ref_plan.ir,
        process=OpSpec.from_callable(lambda p, r: 0),  # closure -> opaque
        combine=ref_plan.combine,
        empty=ref_plan.empty,
        partitions=ref_plan.partitions,
    )
    assert opaque_plan.task_id(part) != ref_plan.task_id(part)
    back = DurablePlan.from_bytes(opaque_plan.to_bytes())
    assert back.opaque and back.process.kind == "opaque"
    assert back.to_bytes() == opaque_plan.to_bytes()


def test_malformed_op_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="module:qualname"):
        OpSpec.from_ref("no_colon_here").resolve()
    with pytest.raises(ValueError, match="unknown OpSpec kind"):
        OpSpec._from_json({"kind": "bogus"})


# ---- runs with no user source files -------------------------------------------------------------
def test_plan_resolves_and_runs_with_no_user_source_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # embed an opaque closure by value, then load + run it in a FRESH interpreter whose working dir
    # and import path contain none of this test's source — proving no source-file dependence.
    factor = 3

    def scale(x: int) -> int:
        return x * factor

    plan = DurablePlan(
        ir=_ir(),
        process=OpSpec.from_callable(scale),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:int"),
    )
    blob_path = tmp_path / "plan.bin"
    blob_path.write_bytes(plan.to_bytes())

    child = (
        "import sys; from graphed_core import DurablePlan;"
        "p=DurablePlan.from_bytes(open(sys.argv[1],'rb').read());"
        "g=p.graph();"  # the IR rebuilds with no source files
        "print(p.process.resolve()(14), g.node_count())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child, str(blob_path)],
        cwd=tmp_path,  # nowhere near this test's directory
        env={"PATH": "/usr/bin:/bin"},  # scrub PYTHONPATH; rely only on installed packages
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["42", "3"], proc.stdout  # 42 = 14*3; 3 IR nodes
