"""M8 integration — the "compile once, run on N datasets" deployment pattern, end to end.

A realistic multi-observable analysis is compiled ONCE to an optimized, interned ``DurablePlan``
(``deployment.compile_plan``) and re-targeted at several synthetic datasets via
``DurablePlan.for_dataset``, all sharing ONE checkpoint ``Store``. These tests exercise the pattern
for **correctness** (per-dataset bit-for-bit, partition-count invariance), **responsiveness**
(kill mid-deployment then resume, skipping completed work; a finished deployment redoes nothing),
and **usability/performance** (one compiled artifact reused across datasets; retargeting is cheap and
content-addressing namespaces datasets in a single store). It is the regression guard for the whole
deployment story.
"""

from __future__ import annotations

import time

import deployment
import numpy as np
import pytest

from graphed.checkpoint import Store, run_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt

CHUNK = 700  # N_EVENTS=1800 -> 3 chunks per dataset


def _run_deployment(plan, datasets, store, **kw):  # type: ignore[no-untyped-def]
    """The deployment idiom under test: one compiled plan, one store, run over each dataset."""
    return {ds.uri: run_resumable(plan.for_dataset(ds, chunk_size=CHUNK), store, **kw) for ds in datasets}


# ---- the compiled artifact is realistic + reused ------------------------------------------------
def test_compiled_plan_is_a_realistic_optimized_graph() -> None:
    unreduced, reduced = deployment.graph_complexity()
    # a genuine multi-observable analysis (incl. a dimuon-mass combinatoric sub-DAG) ...
    assert unreduced >= 40, f"recorded graph too small to be representative ({unreduced} nodes)"
    # ... reduced to a concise set of stage-nodes (graphed's whole thesis), but still non-trivial
    assert reduced < unreduced and reduced >= 12
    assert deployment.compile_plan().graph().node_count() == reduced


def test_compile_once_artifact_is_shared_across_datasets() -> None:
    plan = deployment.compile_plan()
    targeted = [plan.for_dataset(ds, chunk_size=CHUNK) for ds in deployment.DATASETS]
    assert all(p.ir is plan.ir for p in targeted)  # no recompile per dataset
    assert {p.ir_fingerprint() for p in targeted} == {plan.ir_fingerprint()}  # one computation


def test_retargeting_is_cheap() -> None:
    plan = deployment.compile_plan()  # cached compile
    t0 = time.perf_counter()
    for _ in range(200):
        plan.for_dataset(deployment.DATASETS[0], chunk_size=CHUNK)
    # retargeting is pure metadata work (no recording/optimizing) -> sub-second for 200 deployments
    assert time.perf_counter() - t0 < 5.0


# ---- correctness over a single and many datasets ------------------------------------------------
def test_single_dataset_deployment_matches_single_pass(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = deployment.DATASETS[0]
    r = run_resumable(deployment.compile_plan().for_dataset(ds, chunk_size=CHUNK), Store(tmp_path))
    assert np.array_equal(r.value, deployment.reference_for(ds.uri))
    assert r.report.executed == 3 and r.report.skipped == 0


def test_many_datasets_share_one_store_with_correct_per_dataset_results(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan, store = deployment.compile_plan(), Store(tmp_path)
    results = _run_deployment(plan, deployment.DATASETS, store)
    for ds in deployment.DATASETS:
        assert np.array_equal(results[ds.uri].value, deployment.reference_for(ds.uri)), ds.uri
    # one store holds every dataset's tasks without collision: 3 datasets x 3 chunks, all distinct
    assert len(store.completed()) == 3 * 3


def test_datasets_produce_distinct_results() -> None:
    # the deployment really is running on different inputs (not accidentally the same data)
    refs = [deployment.reference_for(ds.uri) for ds in deployment.DATASETS]
    assert not any(np.array_equal(refs[0], r) for r in refs[1:])


# ---- responsiveness: kill mid-deployment, then resume -------------------------------------------
def test_kill_mid_deployment_then_resume_is_correct_and_incremental(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan, store = deployment.compile_plan(), Store(tmp_path)
    ds0, ds1, ds2 = deployment.DATASETS

    # first dataset finishes; the second is killed after 2 of its 3 chunks commit
    run_resumable(plan.for_dataset(ds0, chunk_size=CHUNK), store)
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan.for_dataset(ds1, chunk_size=CHUNK), store, _kill_after=2)

    # resume the WHOLE deployment on the same store
    results = _run_deployment(plan, deployment.DATASETS, store)
    rep = {uri: r.report for uri, r in results.items()}
    assert rep[ds0.uri].executed == 0 and rep[ds0.uri].skipped == 3  # already done -> all reused
    assert rep[ds1.uri].skipped == 2 and rep[ds1.uri].executed == 1  # only the unfinished chunk runs
    assert rep[ds2.uri].executed == 3  # never started before -> fresh
    for ds in deployment.DATASETS:
        assert np.array_equal(results[ds.uri].value, deployment.reference_for(ds.uri))


def test_finished_deployment_redoes_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan, store = deployment.compile_plan(), Store(tmp_path)
    _run_deployment(plan, deployment.DATASETS, store)
    again = _run_deployment(plan, deployment.DATASETS, store)
    assert all(r.report.executed == 0 and r.report.skipped == 3 for r in again.values())
    for ds in deployment.DATASETS:
        assert np.array_equal(again[ds.uri].value, deployment.reference_for(ds.uri))


def test_deployment_is_partition_count_invariant(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = deployment.DATASETS[1]
    ref = deployment.reference_for(ds.uri)
    plan = deployment.compile_plan()
    for i, cs in enumerate((1800, 600, 173)):  # whole, even, ragged chunking
        r = run_resumable(plan.for_dataset(ds, chunk_size=cs), Store(tmp_path / str(i)))
        assert np.array_equal(r.value, ref), f"chunk_size={cs} changed the result"
