"""Fixtures for the frontend m53 policy suite — the `composes_as_union`, `points=` and
`max_universes` knobs on `graphed.vary`.

Numpy-idiom and awkward-free: the REQUIRED free-threaded CI job collects `tests/frozen/frontend`
whole with only `pytest hypothesis numpy` installed, so nothing here may import `graphed.awkward`,
`awkward`, `hist`, `boost_histogram`, `pyarrow` or `pandas` at module OR body level.

The b-tag SF members are read off the jes-varied `pt`, so each depends on `jes` and a plain
`graphed.vary` fans out the full jes(3) x btag(5) grid. Every m53-new symbol — the auto-fanout, the
new keywords — is reached only inside the functions the tests call, so the tree COLLECTS against a
tree with no m53 implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import graphed
from graphed import Session
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record

VECTOR = np.arange(1.0, 13.0)
JES_UP, JES_DOWN = 1.10, 0.90

#: the four b-tag SF tags, each read off the jes-varied pt → all four depend on `jes`
BTAG_TAGS = ("hf_up", "hf_down", "lf_up", "lf_down")
BTAG_FACTOR = {"hf_up": 1.03, "hf_down": 0.97, "lf_up": 1.05, "lf_down": 0.95}

#: the pre-m53 union (seven the collapse produced) and the eight machine-named joints
UNION_LABELS = ("nominal", "jes_up", "jes_down", *(f"btag_{tag}" for tag in BTAG_TAGS))
JOINT_LABELS = tuple(
    f"btag_{tag}__jes_{d}" for tag in BTAG_TAGS for d in ("up", "down")
)

#: every label the full grid mints -> its point (the jes(3) x btag(5) = 15 universes)
GRID_POINTS: dict[str, dict[str, str]] = {
    "nominal": {},
    "jes_up": {"jes": "up"},
    "jes_down": {"jes": "down"},
    **{f"btag_{tag}": {"btag": tag} for tag in BTAG_TAGS},
    **{
        f"btag_{tag}__jes_{d}": {"btag": tag, "jes": d}
        for tag in BTAG_TAGS
        for d in ("up", "down")
    },
}


def _jes_context() -> tuple[Session, EventContext]:
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR, eta=VECTOR / 10.0)
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt, "eta": record["eta"]})
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {"up": pt * JES_UP, "down": pt * JES_DOWN}}
    )
    return session, shifted


def fanout_weight(*, points: Any = None, **vary_kwargs: Any) -> tuple[Session, EventContext, Any]:
    """A jes-dependent b-tag weight family: jes(3) x btag(5) = 15. ``points`` (placement entries)
    merge into the unified ``variations=`` list; a pure-declare call (``points is None``) keeps the
    dict channel. Extra keywords (``composes_as_union=``, ``max_universes=``) pass through."""
    session, shifted = _jes_context()
    pt = shifted["pt"]  # jes-varied
    central = pt * 1.0
    members = {tag: pt * factor for tag, factor in BTAG_FACTOR.items()}
    variations: Any = members if points is None else [*members.items(), *points]
    registered = graphed.vary(
        shifted, "btag", central, is_weight=True, variations=variations, **vary_kwargs
    )
    return session, registered, graphed.weight(registered)


def numeric_fanout(*, points: Any = None, **vary_kwargs: Any) -> EventContext:
    """A jes-dependent weight family over NUMERICALLY tagged jes universes ('2' / '0p5'), for a
    precision (numeric-coordinate) placement. ``points`` merges into the ``variations=`` list."""
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR)
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt})
    shifted = graphed.vary(ctx, "jes", collections={"pt": {"2": pt * 1.2, "0p5": pt * 0.8}})
    spt = shifted["pt"]  # jes-varied, numeric tags '2' / '0p5'
    members = {"2": spt * 1.1}
    variations: Any = members if points is None else [*members.items(), *points]
    return graphed.vary(
        shifted, "muF", spt * 1.0, is_weight=True, variations=variations, **vary_kwargs
    )
