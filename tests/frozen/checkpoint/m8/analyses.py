"""Picklable analysis glue for the M8 checkpoint suite.

All process/combine/empty callables are **module-level** (so a ``DurablePlan`` can reference them by
import path and resolve them on a source-free machine), and all are deterministic (so a resumed run
matches an uninterrupted run bit-for-bit). A small representative IR is attached to each plan so the
content-addressed ``task_id`` is meaningful.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from graphed_core import DurablePlan, GraphStore, OpSpec, Partition

N_VALUES = 6000
SEED = 20260605
BINS = 20
LO, HI = 0.0, 100.0


# ---- a deterministic global dataset; a partition is a contiguous slice of it --------------------
def _all_values() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.uniform(LO, HI, size=N_VALUES)


def _hist(values: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=BINS, range=(LO, HI))[0].astype(np.int64)


# ---- reduction spec (all picklable, all importable by ref) --------------------------------------
def histogram_chunk(partition: Partition, resources: Any) -> np.ndarray:
    vals = _all_values()[partition.entry_start : partition.entry_stop]
    return _hist(vals)


def oom_if_big(partition: Partition, resources: Any) -> np.ndarray:
    # simulate a partition too large to process whole: succeeds only once split small enough
    if partition.n_entries > 800:
        raise MemoryError(f"OOM on {partition.n_entries} entries")
    return histogram_chunk(partition, resources)


def flaky_twice(partition: Partition, resources: Any) -> np.ndarray:
    # transient failure: fails until it has been attempted twice for this partition (state lives in
    # the resources dict, which RetryN reuses across attempts)
    counts = resources.setdefault("attempts", {})  # type: ignore[union-attr]
    key = (partition.entry_start, partition.entry_stop)
    counts[key] = counts.get(key, 0) + 1
    if counts[key] < 2:
        raise RuntimeError("transient worker hiccup")
    return histogram_chunk(partition, resources)


def always_fail(partition: Partition, resources: Any) -> np.ndarray:
    raise RuntimeError(f"always fails on {partition.entry_start}:{partition.entry_stop}")


def fails_on_bad_worker(partition: Partition, resources: Any) -> np.ndarray:
    # models a flaky worker: fails while running on the "bad" context, succeeds once moved elsewhere
    if resources and resources.get("bad"):
        raise RuntimeError("bad worker")
    return histogram_chunk(partition, resources)


_POISON_START = 1000


def poison_one(partition: Partition, resources: Any) -> np.ndarray:
    # raise a source-mapped StageError (M6) on a specific partition, to exercise dead-letter
    # provenance harvesting; every other partition succeeds normally
    if partition.entry_start == _POISON_START:
        from graphed_debug import SourceFrame, StageError  # noqa: PLC0415

        raise StageError(
            op="jet_pt",
            frames=(SourceFrame("analysis.py", 42, "select", "events.Jet.pt[mask]"),),
            input_forms=("var * float64",),
            partition=f"{partition.uri}@{partition.entry_start}:{partition.entry_stop}",
            cause_type="ValueError",
            cause_message="poison partition",
            opt_level=1,
        )
    return histogram_chunk(partition, resources)


def hist_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def hist_empty() -> np.ndarray:
    return np.zeros(BINS, dtype=np.int64)


# ---- plan construction --------------------------------------------------------------------------
def _ir() -> bytes:
    g = GraphStore()
    src = g.add_source("events", {"uri": "corpus://values"})
    pt = g.add_op("value", [src])
    cut = g.add_op("gt", [pt], {"thr": 0.0})
    out = g.add_reduction("hist", [cut], {"bins": BINS})
    return g.serialize(outputs=[out])  # [freeze-M22-1: mark_output removed; outputs per request]


def partitions(n_chunks: int) -> tuple[Partition, ...]:
    edges = np.linspace(0, N_VALUES, n_chunks + 1, dtype=int)
    return tuple(
        Partition("corpus://values", "Events", int(edges[i]), int(edges[i + 1])) for i in range(n_chunks)
    )


def build_plan(process_ref: str, n_chunks: int, *, error_budget: int | None = None) -> DurablePlan:
    return DurablePlan(
        ir=_ir(),
        process=OpSpec.from_ref(process_ref),
        combine=OpSpec.from_ref("analyses:hist_add"),
        empty=OpSpec.from_ref("analyses:hist_empty"),
        partitions=partitions(n_chunks),
        read_columns=("value",),
        stopping={} if error_budget is None else {"error_budget": error_budget},
    )


def reference() -> np.ndarray:
    """The single-pass histogram of the whole dataset (the bit-for-bit target)."""
    return _hist(_all_values())


# ---- a realistic HEP analysis recorded through graphed + executed per chunk ----------------------
N_EVENTS_HEP = 3000
SEED_HEP = 7
HEP_BINS, HEP_LO, HEP_HI = 20, 0.0, 200.0


def _met_counts(values: object) -> np.ndarray:
    from graphed_corpus.histograms import hist1d  # noqa: PLC0415

    h = hist1d(values, bins=HEP_BINS, start=HEP_LO, stop=HEP_HI, name="MET")
    return np.asarray(h.values(), dtype=np.int64)


def met_chunk(partition: Partition, resources: Any) -> np.ndarray:
    """Record + execute the ADL-q1 MET-pt histogram on one event chunk (a real graphed analysis)."""
    from graphed import Session  # noqa: PLC0415
    from graphed_awkward import AwkwardBackend, from_awkward  # noqa: PLC0415
    from graphed_corpus import make_events  # noqa: PLC0415

    events = make_events(n_events=N_EVENTS_HEP, seed=SEED_HEP)[partition.entry_start : partition.entry_stop]
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    return _met_counts(s.materialize(ev.MET.pt))


def met_reference() -> np.ndarray:
    from graphed import Session  # noqa: PLC0415
    from graphed_awkward import AwkwardBackend, from_awkward  # noqa: PLC0415
    from graphed_corpus import make_events  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", make_events(n_events=N_EVENTS_HEP, seed=SEED_HEP))
    return _met_counts(s.materialize(ev.MET.pt))


def met_partitions(n_chunks: int) -> tuple[Partition, ...]:
    edges = np.linspace(0, N_EVENTS_HEP, n_chunks + 1, dtype=int)
    return tuple(
        Partition("corpus://events", "Events", int(edges[i]), int(edges[i + 1])) for i in range(n_chunks)
    )


def build_met_plan(n_chunks: int) -> DurablePlan:
    return DurablePlan(
        ir=_ir(),
        process=OpSpec.from_ref("analyses:met_chunk"),
        combine=OpSpec.from_ref("analyses:hist_add"),
        empty=OpSpec.from_ref("analyses:hist_empty"),
        partitions=met_partitions(n_chunks),
        read_columns=("MET_pt",),
    )
