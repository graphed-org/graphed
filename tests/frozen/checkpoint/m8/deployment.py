"""A realistic "compile once, run on N datasets" deployment, for the M8 integration tests.

A multi-observable HEP analysis (MET, HT, jet/muon multiplicities, leading-object pts) is recorded
through the graphed frontend **once** and serialized to an **optimized, interned** durable IR (the
"compile" step). The same compiled `DurablePlan` is then re-targeted at several **synthetic
datasets** (distinct per uri) via `DurablePlan.for_dataset`, and run through the checkpoint runner.

All process/combine/empty callables are module-level so the plan can reference them by import path.
Each dataset's events are derived deterministically from its uri, so per-dataset results are distinct
and independently reproducible.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any

import numpy as np

from graphed.core import Dataset, DurablePlan, OpSpec

N_EVENTS = 1800

# (name, bins, lo, hi) for each observable; the per-chunk partial is the concatenation of these.
_AXES: list[tuple[str, int, float, float]] = [
    ("met", 20, 0.0, 200.0),
    ("ht", 20, 0.0, 600.0),
    ("njet", 10, 0.0, 10.0),
    ("nmu", 6, 0.0, 6.0),
    ("sum_mu_pt", 20, 0.0, 200.0),
    ("first_jet_pt", 20, 0.0, 300.0),
    ("dimuon_mass", 24, 0.0, 200.0),
]
TOTAL_BINS = sum(b for _, b, _, _ in _AXES)

DATASETS = [Dataset(f"corpus://ttbar_{i}", n_events=N_EVENTS, name=f"ttbar_{i}") for i in range(3)]


# ---- per-dataset synthetic events (deterministic from the uri) -----------------------------------
def _seed_for(uri: str) -> int:
    return int.from_bytes(hashlib.sha256(uri.encode()).digest()[:4], "big")


def _events_for(uri: str) -> Any:
    from graphed_corpus import make_events  # noqa: PLC0415

    return make_events(n_events=N_EVENTS, seed=_seed_for(uri))


# ---- the analysis graph (recorded through the frontend) ------------------------------------------
def _pair_mass(o1: Any, o2: Any) -> Any:
    """Invariant mass of an object pair (the dimuon-mass calc — a deep arithmetic sub-DAG)."""
    px = o1.pt * np.cos(o1.phi) + o2.pt * np.cos(o2.phi)
    py = o1.pt * np.sin(o1.phi) + o2.pt * np.sin(o2.phi)
    pz = o1.pt * np.sinh(o1.eta) + o2.pt * np.sinh(o2.eta)
    e = np.sqrt(o1.pt**2 * np.cosh(o1.eta) ** 2 + o1.mass**2) + np.sqrt(
        o2.pt**2 * np.cosh(o2.eta) ** 2 + o2.mass**2
    )
    return np.sqrt(np.maximum(e**2 - (px**2 + py**2 + pz**2), 0.0))


def _record_observables(ev: Any) -> list[Any]:
    """Record the multi-observable analysis on a graphed Array ``ev``; return the observable Arrays
    in ``_AXES`` order. Shared object selections (jets, muons) intern once, and the dimuon-mass calc
    adds a deep arithmetic sub-DAG -> a graph of realistic analysis complexity."""
    from graphed.awkward import gak  # noqa: PLC0415

    jets = ev.Jet[ev.Jet.pt > 30]
    mu = ev.Muon[ev.Muon.pt > 10]
    pairs = gak.combinations(mu, 2, fields=["a", "b"])
    return [
        ev.MET.pt,
        gak.sum(jets.pt, axis=1),
        gak.num(jets, axis=1),
        gak.num(mu, axis=1),
        gak.sum(mu.pt, axis=1),
        gak.fill_none(gak.firsts(jets.pt), 0.0),
        gak.fill_none(gak.firsts(_pair_mass(pairs.a, pairs.b)), 0.0),
    ]


def _counts(values: Any, bins: int, lo: float, hi: float) -> np.ndarray:
    from graphed_corpus.histograms import hist1d  # noqa: PLC0415

    return np.asarray(hist1d(values, bins=bins, start=lo, stop=hi, name="x").values(), dtype=np.int64)


def _concat(materialized: list[Any]) -> np.ndarray:
    parts = [_counts(v, b, lo, hi) for (_, b, lo, hi), v in zip(_AXES, materialized, strict=True)]
    return np.concatenate(parts)


# ---- reduction spec (referenced by import path) -------------------------------------------------
def analysis_chunk(partition: Any, resources: Any) -> np.ndarray:
    """Execute the analysis on one chunk of the dataset selected by ``partition.uri``."""
    from graphed import Session  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415

    events = _events_for(partition.uri)[partition.entry_start : partition.entry_stop]
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    return _concat([s.materialize(o) for o in _record_observables(ev)])


def hist_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def hist_empty() -> np.ndarray:
    return np.zeros(TOTAL_BINS, dtype=np.int64)


# ---- compile once -------------------------------------------------------------------------------
@functools.cache
def compiled_ir() -> bytes:
    """Record the analysis once and serialize the OPTIMIZED, interned graph (the durable artifact).

    Cached: the compile step happens once per process, mirroring the deployment pattern (and keeping
    the suite responsive)."""
    from graphed import Session  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", _events_for("template://compile"))  # structure only, not data
    return s.serialized_ir(*_record_observables(ev), optimize=True)


@functools.cache
def graph_complexity() -> tuple[int, int]:
    """(unreduced, optimized) node counts of the recorded analysis — for the complexity assertion."""
    from graphed import Session  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415
    from graphed.core import GraphStore  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", _events_for("template://compile"))
    obs = _record_observables(ev)
    unreduced = GraphStore.deserialize(s.serialized_ir(*obs, optimize=False)).node_count()
    reduced = GraphStore.deserialize(s.serialized_ir(*obs, optimize=True)).node_count()
    return unreduced, reduced


def compile_plan() -> DurablePlan:
    """The compiled analysis as a partition-less DurablePlan, ready to target datasets."""
    return DurablePlan(
        ir=compiled_ir(),
        process=OpSpec.from_ref("deployment:analysis_chunk"),
        combine=OpSpec.from_ref("deployment:hist_add"),
        empty=OpSpec.from_ref("deployment:hist_empty"),
        read_columns=("MET_pt", "Jet_pt", "Muon_pt"),
    )


@functools.cache
def reference_for(uri: str) -> np.ndarray:
    """The single-pass result for one dataset (the bit-for-bit target for a chunked deployment)."""
    from graphed import Session  # noqa: PLC0415
    from graphed.awkward import AwkwardBackend, from_awkward  # noqa: PLC0415

    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", _events_for(uri))
    return _concat([s.materialize(o) for o in _record_observables(ev)])
