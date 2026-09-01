"""Fixtures for the m48 event-context, idiom and dispatch suite.

Every anchor in this tree imports `graphed.awkward` — the gak enumerations and representatives, the
whole §2.6 event-context family, and §4.1's correctionlib recording, which yields a payload
descriptor only under `AwkwardBackend` (§10/m48 partition rule (2)). The two numpy-idiom fixtures at
the bottom are duplicated rather than shared: `tests/frozen/frontend/m48` must stay importable under
`pytest hypothesis numpy` alone, so no helper crosses between the two trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import awkward as ak
import numpy as np
from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.core import Partition
from graphed.numpy import NumpyBackend, NumpyForm, from_array, from_record

#: one synthetic dataset for the whole tree; jagged Jet/Photon/Muon/Electron + a per-event MET
EVENTS = make_events(n_events=200, seed=48)

#: §2.3e's frozen operand-KIND vocabulary — `src` argument fixtures name one of these, and the
#: frozen test owns one CONTEXTED array of each, all read through the same context
OPERAND_KINDS = ("flat", "jagged", "record", "mask", "option")


def awkward_session() -> tuple[Session, Array]:
    """A session over the corpus events; the returned `Array` is the root event RECORD."""
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", EVENTS)


def events_context(*, is_data: bool = False) -> tuple[Session, Any]:
    """The §2.6 event context. `is_data=True` is the explicit constructor flag §2.6d guards on."""
    session, root = awkward_session()
    return session, ga.gnano.events(root, is_data=is_data)


def reserved_names_context() -> tuple[Session, Any]:
    """A context whose tree literally carries branches named `collections`, `weights` and `vary`.

    §2.6a: the context reserves NO names — branch names are analysis-controlled — and §2.1's
    `collections={"collections": ...}` self-reference has no other operand.
    """
    session, root = awkward_session()
    tree = gak.with_field(root, root.Jet, "collections")  # a COLLECTION of that name (§2.1)
    tree = gak.with_field(tree, root.MET.pt, "weights")
    tree = gak.with_field(tree, root.MET.pt, "vary")
    return session, ga.gnano.events(tree)


def operands(source: Any) -> dict[str, Array]:
    """One array per §2.3e operand kind, all read through `source` so they share its handle.

    graphed type-checks the primary operand at RECORD time through the backend's `op_form`, so one
    array cannot serve every gak function — the kind is what an `src` fixture's slot asks for.
    """
    jets = source.Jet
    return {
        "flat": source.MET.pt,
        "jagged": jets.pt,
        "record": jets,
        "mask": jets.pt > 25.0,
        "option": gak.firsts(jets.pt),
    }


def shifted_jets(source: Any, factor: float) -> Array:
    """A lockstep JES-shifted Jet RECORD, the corpus's own `ak.with_field` spelling."""
    jets = source.Jet
    return gak.with_field(jets, jets.pt * factor, "pt")


def shifted_met(source: Any, factor: float) -> Array:
    met = source.MET
    return gak.with_field(met, met.pt * factor, "pt")


def jes_kwargs(source: Any) -> dict[str, dict[str, Array]]:
    """The lockstep shift form's collection mappings: Jet and MET move together (§2.6a)."""
    return {
        "Jet": {"up": shifted_jets(source, 1.05), "down": shifted_jets(source, 0.95)},
        "MET": {"up": shifted_met(source, 1.02), "down": shifted_met(source, 0.98)},
    }


def pu_weight(source: Any, scale: float) -> Array:
    """A per-EVENT weight factor read through `source` (so it lives in that row space, §2.1(b))."""
    return gak.full_like(source.MET.pt, 1.0) * scale


def btag_weight(jets: Array, scale: float = 1.0) -> Array:
    """The corpus shape: a per-JET scale factor producted over axis 1 into a per-event weight."""
    return gak.prod(1.0 + jets.btag * scale, axis=1)


def as_list(value: object) -> Any:
    """Materialized values compared elementwise; `ak.to_list` is exact for both idioms."""
    return ak.to_list(value)


def reindexed(session: Session, value: Array, mask: Array) -> Any:
    """The manual re-indexing reference: apply THAT label's own mask to the ancestor value."""
    return as_list(session.materialize(value[mask]))


#: --- numpy-idiom fixtures, for §2.3d's property and string-getitem halves -------------------
#: `NumpyArray.T` raises on a >=2-D partitioned form and on any record form, so the property
#: measurement needs the VECTOR source while the reserved-name control needs the RECORD one.

VECTOR = np.arange(1.0, 13.0)


def vector_source() -> tuple[Session, Array]:
    session = Session(NumpyBackend())
    return session, from_array(session, "x", VECTOR)


def record_source() -> tuple[Session, Array]:
    session = Session(NumpyBackend())
    return session, from_record(session, "r", node_id=np.arange(4), pt=np.arange(1.0, 5.0))


def loose_varied(x: Array, name: str = "jes") -> graphed.Varied:
    """§2.1(a)'s loose primitive, the construction path with no context involved."""
    return graphed.vary(x, name, up=x * 1.5, down=x * 0.5)


@dataclass
class ArraySource:
    """A numpy-idiom `PartitionedSource` — `aggregate_plan` binds exactly one per session, so its
    §2.3d positive control ("the same verb on a plain `Array` still works") needs one."""

    data: np.ndarray

    def __call__(self) -> np.ndarray:
        raise AssertionError("the whole-dataset loader must never run during a plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://vector", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Any, columns: Any, resources: Any) -> np.ndarray:
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def partitioned_vector_source() -> tuple[Session, Array]:
    session = Session(NumpyBackend())
    form = NumpyForm(VECTOR.dtype, shape=(None,))
    return session, session.source("x", form=form, data=ArraySource(VECTOR))
