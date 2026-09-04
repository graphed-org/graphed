"""Multiparam correctionlib `External`s must EXECUTE through `aggregate_plan` — the plan path, not
just recording.

`test_correctionlib_multiparam` (frozen) pins the RECORDING side: one payload, N `systematic`
universes as distinct nodes sharing a `content_hash`. Nothing ran them. This suite runs them through
`aggregate_plan` — the engine every histogram plan is built on — and pins the three things the fix
turns on:

* `aggregate_plan` auto-wires EVERY External surviving in the compiled IR from the recording session
  (upstream corrections included), so a plan whose output cone reads a correctionlib node no longer
  raises ``External payload '…' needs an evaluator``;
* execution resolves each External by its ``(content_hash, params)`` key, so N universes off ONE
  payload get N evaluators — they do not collapse onto the shared payload hash;
* the recorded evaluator is PICKLABLE (a module-level `_PluginEvaluator`, not a local closure), so
  the plan ships to a process pool — the distributed "hundreds of histograms" case.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed_corpus import make_events

from graphed import Array, Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources
from graphed.preserve import CORRECTIONLIB_PLUGIN, record_external
from graphed.preserve.externals._base import _PluginEvaluator

pytest.importorskip("correctionlib")

EVENTS = make_events(n_events=1_500, seed=48)
N_PARTITIONS = 4
SYSTEMATICS = ("nominal", "up", "down")

_SF_EDGES = [0.0, 4.0, 5.0, 6.0, 100.0]
_SF_CONTENT = {
    "nominal": [1.0, 1.0, 1.0, 1.0],
    "up": [1.05, 1.10, 1.15, 1.20],
    "down": [0.95, 0.90, 0.85, 0.80],
}


def _correctionlib_json() -> bytes:
    content = {
        syst: {"nodetype": "binning", "input": "x", "edges": _SF_EDGES, "content": vals, "flow": "clamp"}
        for syst, vals in _SF_CONTENT.items()
    }
    cset = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "btag_sf",
                "version": 1,
                "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [{"key": k, "value": v} for k, v in content.items()],
                },
            }
        ],
    }
    return json.dumps(cset, sort_keys=True).encode("utf-8")


@dataclass
class CorpusEvents:
    """A `graphed.write.PartitionedSource` over the corpus events."""

    data: ak.Array
    part_reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("corpus://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def _universes() -> tuple[Session, list[Array]]:
    """Three per-event SF Arrays off ONE correctionlib payload, differing only in `systematic`."""
    session = Session(AwkwardBackend())
    source = CorpusEvents(EVENTS)
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    events = session.source("events", form=form, data=source)
    njet = gak.num(events.Jet[events.Jet.pt > 25], axis=1)
    payload = _correctionlib_json()
    outs = [
        record_external(
            session, CORRECTIONLIB_PLUGIN, payload, [njet], params={"name": "btag_sf", "systematic": syst}
        )
        for syst in SYSTEMATICS
    ]
    return session, outs


def _sums(vals: list[object]) -> np.ndarray:
    # reduce runs at execution time on CONCRETE awkward arrays (not deferred graphed Arrays)
    return np.array([float(ak.sum(v)) for v in vals], dtype="float64")


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def _zeros() -> np.ndarray:
    return np.zeros(len(SYSTEMATICS))


def _totals(executor: Any) -> np.ndarray:
    _session, outs = _universes()
    plan = aggregate_plan(*outs, reduce=_sums, combine=_add, empty=_zeros, steps_per_file=N_PARTITIONS)
    return executor.run(plan).value


def test_multiparam_universes_execute_to_distinct_totals() -> None:
    """One payload hash, three params → three evaluators. A collision onto the shared hash would tie
    the totals; `up`/`down` bracket nominal because the SF was actually read per universe."""
    nominal, up, down = _totals(SequentialRunner())
    assert up > nominal > down
    assert len({round(nominal, 9), round(up, 9), round(down, 9)}) == 3


def test_the_plan_ships_to_a_process_pool() -> None:
    """The recorded correctionlib evaluator must pickle and re-load in a worker (the pre-fix local
    closure raised ``Can't get local object`` on plan submission)."""
    ProcessPoolExecutor = pytest.importorskip("graphed_exec_local").ProcessPoolExecutor
    seq = _totals(SequentialRunner())
    par = _totals(ProcessPoolExecutor(max_workers=2))
    assert np.allclose(par, seq, rtol=1e-12, atol=0)


def test_the_recorded_evaluator_is_a_picklable_object() -> None:
    """`preserve.record_external` records a module-level `_PluginEvaluator`, not a `<locals>` closure,
    and it round-trips through pickle and evaluates identically."""
    session, _outs = _universes()
    (fn, _inputs) = next(iter(session._externals.values()))
    assert isinstance(fn, _PluginEvaluator)
    restored = pickle.loads(pickle.dumps(fn))
    assert isinstance(restored, _PluginEvaluator)
    x = ak.Array([3, 5, 7])
    assert ak.to_list(fn(x)) == ak.to_list(restored(x))
