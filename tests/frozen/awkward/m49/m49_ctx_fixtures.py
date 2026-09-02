"""Fixtures for the m49 awkward-idiom anchors.

Every anchor here imports `graphed.awkward`, so none of them can live in `tests/frozen/frontend/m49`
(a required free-threaded job collects that tree under `pytest hypothesis numpy` alone). The helper
carries an `m49_` prefix even though `run-tests.sh` gives `tests/frozen/awkward/m49` its own process:
prepend import mode binds a bare top-level helper name to whichever sibling directory imported first.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, from_awkward, gak

#: one synthetic dataset for the whole tree; jagged Jet/Photon/Muon/Electron + a per-event MET
EVENTS = make_events(n_events=200, seed=49)

#: a second per-event regular structure, deliberately NOT the corpus's, for the broadcast anchor
REGULAR_VALUE = np.arange(600.0).reshape(200, 3)
REGULAR_FACTOR = np.arange(1000.0).reshape(200, 5)
REGULAR_COMPATIBLE = REGULAR_VALUE * 0.5

#: a jagged factor whose per-row counts differ from the corpus Jet counts EVERYWHERE (one more per
#: row), at a distinguishable dtype: the blame message has to be checkable against the FACTOR's own
#: structure, which a `var * float64` twin of the value could not do.
_JAGGED_COUNTS = np.asarray(ak.num(EVENTS.Jet, axis=1)) + 1
JAGGED_FACTOR = ak.unflatten(np.arange(int(_JAGGED_COUNTS.sum()), dtype=np.int64), _JAGGED_COUNTS)


def awkward_session() -> tuple[Session, Array]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", EVENTS)


def events_context() -> tuple[Session, Any]:
    """The §2.6 event context over the corpus events."""
    session, root = awkward_session()
    return session, ga.gnano.events(root)


def jes_collections(source: Any) -> dict[str, dict[str, Array]]:
    """The lockstep shift form's mappings: Jet and MET move together under one tag set (§2.6a)."""
    jets, met = source.Jet, source.MET
    return {
        "Jet": {"up": gak.with_field(jets, jets.pt * 1.05, "pt"), "down": gak.with_field(jets, jets.pt * 0.95, "pt")},
        "MET": {"up": gak.with_field(met, met.pt * 1.02, "pt"), "down": gak.with_field(met, met.pt * 0.98, "pt")},
    }


def jet_weight(source: Any, scale: float = 1.0) -> Array:
    """A per-event weight factor whose cone reaches the JET collection (the corpus b-tag shape)."""
    return gak.prod(1.0 + source.Jet.btag * scale, axis=1)


def met_weight(source: Any, scale: float = 1.0) -> Array:
    """A per-event weight factor whose cone reaches the MET collection and nothing else."""
    return source.MET.pt * scale


def weight_universes(ctx: Any) -> list[Array]:
    """The ambient registry's members, one marked output per label (`compile_ir` refuses a Varied)."""
    ambient = graphed.weight(ctx)
    return [graphed.universe(ambient, label) for label in graphed.labels(ambient)]


def inner_type(data: Any) -> str:
    """A structure's type below its outer length — the part a typetracer form and a materialized
    partition render identically, so a blame message can be checked against it on either path."""
    return str(ak.Array(data).type).split(" * ", 1)[1]


def as_list(value: object) -> Any:
    return ak.to_list(value)
