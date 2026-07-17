"""M8 — the "compile once, run on N datasets" deployment primitives.

``DurablePlan.with_partitions`` / ``for_dataset`` / ``for_datasets`` + the ``Dataset`` → partitions
builders let a user compile an analysis once and re-target it at many inputs without re-recording or
re-optimizing the graph. These tests pin: correct chunking, that the computation (the optimized
interned IR) is **shared unchanged** across re-targetings, and that per-dataset ``task_id``s are
disjoint so a single checkpoint store namespaces datasets safely.
"""

from __future__ import annotations

import pytest

from graphed.core import (
    Dataset,
    DurablePlan,
    GraphStore,
    OpSpec,
    Partition,
    partition_dataset,
    partition_datasets,
)


def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "ds"})
    out = g.add_reduction("hist", [g.add_op("pt", [src])])
    return g.serialize(outputs=[out])


def _compiled_plan() -> DurablePlan:
    # a plan with NO partitions yet — the "compiled" analysis, ready to target datasets
    return DurablePlan(
        ir=_ir(),
        process=OpSpec.from_ref("math:hypot"),
        combine=OpSpec.from_ref("operator:add"),
        empty=OpSpec.from_ref("builtins:float"),
        read_columns=("Jet_pt",),
    )


# ---- dataset -> partitions ----------------------------------------------------------------------
def test_partition_dataset_chunks_cover_the_whole_range() -> None:
    ds = Dataset("file://a.root", n_events=100, tree="Events")
    parts = partition_dataset(ds, chunk_size=30)
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 30), (30, 60), (60, 90), (90, 100)]
    assert all(p.uri == "file://a.root" and p.tree == "Events" for p in parts)
    assert sum(p.n_entries for p in parts) == 100  # exact cover, no gaps or overlap


def test_partition_dataset_edge_cases() -> None:
    assert partition_dataset(Dataset("u", 0), chunk_size=10) == ()  # empty dataset
    one = partition_dataset(Dataset("u", 5), chunk_size=100)  # chunk bigger than dataset
    assert len(one) == 1 and one[0].n_entries == 5
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        partition_dataset(Dataset("u", 5), chunk_size=0)


def test_partition_datasets_concatenates() -> None:
    dss = [Dataset("u1", 50), Dataset("u2", 70)]
    parts = partition_datasets(dss, chunk_size=25)
    assert len(parts) == len(partition_dataset(dss[0], chunk_size=25)) + len(
        partition_dataset(dss[1], chunk_size=25)
    )
    assert {p.uri for p in parts} == {"u1", "u2"}


# ---- compile once, retarget many ----------------------------------------------------------------
def test_with_partitions_shares_the_compiled_computation() -> None:
    base = _compiled_plan()
    retargeted = base.with_partitions([Partition("ds-A", "Events", 0, 10)])
    # the optimized interned IR is reused unchanged (no recompile): same object, same fingerprint
    assert retargeted.ir is base.ir
    assert retargeted.ir_fingerprint() == base.ir_fingerprint()
    assert retargeted.process == base.process and retargeted.combine == base.combine
    assert retargeted.read_columns == base.read_columns
    assert tuple(retargeted.partitions) == (Partition("ds-A", "Events", 0, 10),)


def test_for_dataset_and_for_datasets_build_a_deployment() -> None:
    base = _compiled_plan()
    a, b = Dataset("ds-A", 40), Dataset("ds-B", 60)
    pa = base.for_dataset(a, chunk_size=20)
    assert tuple(pa.partitions) == partition_dataset(a, chunk_size=20)
    assert all(p.ir is base.ir for p in (pa,))  # still the same compiled graph

    combined = base.for_datasets([a, b], chunk_size=20)
    assert tuple(combined.partitions) == partition_datasets([a, b], chunk_size=20)


def test_compile_once_is_reused_across_many_datasets() -> None:
    base = _compiled_plan()
    fp = base.ir_fingerprint()
    datasets = [Dataset(f"ds-{i}", 1000) for i in range(5)]
    plans = [base.for_dataset(d, chunk_size=250) for d in datasets]
    # every deployment shares the identical compiled IR (the expensive artifact is built once)
    assert all(p.ir is base.ir and p.ir_fingerprint() == fp for p in plans)


def test_task_ids_are_namespaced_per_dataset() -> None:
    # the SAME analysis over different datasets must produce disjoint task_ids, so one checkpoint
    # store can hold all of them without cross-contamination
    base = _compiled_plan()
    a = base.for_dataset(Dataset("ds-A", 100), chunk_size=50)
    b = base.for_dataset(Dataset("ds-B", 100), chunk_size=50)
    ids_a = {a.task_id(p) for p in a.partitions}
    ids_b = {b.task_id(p) for p in b.partitions}
    assert ids_a.isdisjoint(ids_b)
    assert len(ids_a) == len(a.partitions)  # each chunk a distinct id


def test_retargeting_only_changes_partitions_in_the_serialized_form() -> None:
    base = _compiled_plan().with_partitions([Partition("ds-A", "Events", 0, 10)])
    other = base.with_partitions([Partition("ds-A", "Events", 0, 10)])
    assert base.to_bytes() == other.to_bytes()  # deterministic + identical for identical deployment
    moved = base.with_partitions([Partition("ds-B", "Events", 0, 10)])
    assert moved.to_bytes() != base.to_bytes()  # a different dataset is a different deployment
    assert moved.ir_fingerprint() == base.ir_fingerprint()  # ...but the same computation
