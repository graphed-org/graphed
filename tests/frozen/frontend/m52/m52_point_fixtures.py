"""Awkward-free numpy-idiom programs for the m52 nuisance-point suite.

ci.yml's REQUIRED free-threaded job collects `tests/frozen/frontend` WHOLE with only
`pytest hypothesis numpy` installed, so nothing reachable from this tree may import
`graphed.awkward`, `awkward`, `hist`, `boost_histogram`, `pyarrow` or `pandas` — at module level or
in a body. Event contexts are therefore constructed directly, the `frontend/m49` idiom.

The `m52_` prefix is load-bearing: the whole `tests/frozen/frontend` tree collects in one process
under prepend import mode, where a shared bare helper name binds to whichever sibling directory
imported first.

No m52-new symbol (`graphed._points`, `graphed.points`, the `points=` keyword) is reached at import
time from this module — the keyword appears only inside functions the tests call, so the suite
COLLECTS against a tree with no m52 implementation and fails at RUN time.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

import graphed
from graphed import Array, Session, Varied
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record
from graphed.varied import rebuild

#: the 1-D payload every recording fixture reads
VECTOR = np.arange(1.0, 13.0)

#: distinct per-universe factors, so no two universes of a program materialize alike
JES_UP, JES_DOWN = 1.10, 0.90
BTAG_UP, BTAG_DOWN = 1.20, 0.80
MU_UP, MU_DOWN = 1.05, 0.95
JOINT_FACTOR = 1.37

#: the cuts the selection-carrier context is built on
NOMINAL_CUT, LOOSE_CUT = 4.0, 2.0


def point_api() -> Any:
    """`graphed._points`, resolved at CALL time. C1's module must not be reached at import, or this
    tree stops collecting against a tree that has no m52 implementation."""
    return importlib.import_module("graphed._points")


def source() -> tuple[Session, Array]:
    """A session over a two-field record. The `points` field is deliberate: the shift form takes a
    COLLECTION named `points` through `collections=`, which 2.11 pins."""
    session = Session(NumpyBackend())
    return session, from_record(session, "ev", pt=VECTOR, eta=VECTOR / 10.0, points=VECTOR * 3.0)


def plain_context() -> tuple[Session, EventContext]:
    session, record = source()
    pt = record["pt"]
    return session, EventContext(
        session, pt, collections={"pt": pt, "eta": record["eta"], "points": record["points"]}
    )


# ---- the six existing shapes 2.1 quantifies over -------------------------------------------------
def loose_family() -> tuple[Session, Varied]:
    """The loose form: `vary` on a bare `Array`."""
    session, record = source()
    pt = record["pt"]
    return session, graphed.vary(pt, "jes", up=pt * JES_UP, down=pt * JES_DOWN)


def loose_family_extension() -> tuple[Session, Varied]:
    """A second `vary` call extending an existing family with a further tag."""
    session, base = loose_family()
    nominal = graphed.nominal(base)
    return session, graphed.vary(base, "jes", up2=nominal * (JES_UP**2))


def weight_context() -> tuple[Session, EventContext]:
    session, ctx = plain_context()
    weight = ctx["pt"] * 0.5
    varied = graphed.vary(
        ctx,
        "btag",
        weight,
        is_weight=True,
        variations={"up": weight * BTAG_UP, "down": weight * BTAG_DOWN},
    )
    return session, varied


def shift_context() -> tuple[Session, EventContext]:
    session, ctx = plain_context()
    pt = ctx["pt"]
    varied = graphed.vary(
        ctx, "jes", collections={"pt": {"up": pt * JES_UP, "down": pt * JES_DOWN}}
    )
    return session, varied


def lockstep_shift_context() -> tuple[Session, EventContext]:
    """One family moving two collections together — `_check_lockstep`'s shape."""
    session, ctx = plain_context()
    pt, eta = ctx["pt"], ctx["eta"]
    varied = graphed.vary(
        ctx,
        "jes",
        collections={
            "pt": {"up": pt * JES_UP, "down": pt * JES_DOWN},
            "eta": {"up": eta * 1.01, "down": eta * 0.99},
        },
    )
    return session, varied


def stacked_weight_context() -> tuple[Session, EventContext]:
    session, first = weight_context()
    ambient = graphed.weight(first)
    varied = graphed.vary(
        first,
        "mu",
        ambient,
        is_weight=True,
        variations={"up": ambient * MU_UP, "down": ambient * MU_DOWN},
    )
    return session, varied


#: `{builder: {label: {nuisance: coordinate}}}` — the default point of every label these six mint.
DEFAULT_POINT_MAPS: dict[str, dict[str, dict[str, str]]] = {
    "loose_family": {
        "nominal": {},
        "jes_up": {"jes": "up"},
        "jes_down": {"jes": "down"},
    },
    "loose_family_extension": {
        "nominal": {},
        "jes_up": {"jes": "up"},
        "jes_down": {"jes": "down"},
        "jes_up2": {"jes": "up2"},
    },
    "weight_context": {
        "nominal": {},
        "btag_up": {"btag": "up"},
        "btag_down": {"btag": "down"},
    },
    "shift_context": {
        "nominal": {},
        "jes_up": {"jes": "up"},
        "jes_down": {"jes": "down"},
    },
    "lockstep_shift_context": {
        "nominal": {},
        "jes_up": {"jes": "up"},
        "jes_down": {"jes": "down"},
    },
    "stacked_weight_context": {
        "nominal": {},
        "btag_up": {"btag": "up"},
        "btag_down": {"btag": "down"},
        "mu_up": {"mu": "up"},
        "mu_down": {"mu": "down"},
    },
}

EXISTING_SHAPES = tuple(DEFAULT_POINT_MAPS)


def build(name: str) -> tuple[Session, Any]:
    """Run one of `EXISTING_SHAPES` by name, so a parametrized test names its program in the id."""
    return globals()[name]()


# ---- the §8-g shape: a container whose LABELS outrun its `_tags` ---------------------------------
def shift_then_weight_context() -> tuple[Session, EventContext]:
    """A shift family, then a weight family. The resulting ambient weight carries `jes_up` while its
    `_tags` has no `jes` key at all — §8-g, which is why an axis set comes from the registry."""
    session, shifted = shift_context()
    weight = shifted["pt"] * 0.5
    varied = graphed.vary(
        shifted,
        "btag",
        weight,
        is_weight=True,
        variations={"up": weight * BTAG_UP, "down": weight * BTAG_DOWN},
    )
    return session, varied


# ---- two registered axes, the smallest carrier of a legal TWO-coordinate point --------------------
def two_axis_loose() -> tuple[Session, Varied]:
    """A loose `Varied` over `jes` and `jer` — the loose form's own carrier is `target._tags`."""
    session, record = source()
    pt = record["pt"]
    first = graphed.vary(pt, "jes", up=pt * JES_UP, down=pt * JES_DOWN)
    nominal = graphed.nominal(first)
    return session, graphed.vary(first, "jer", up=nominal * 1.01, down=nominal * 0.99)


def two_axis_context() -> tuple[Session, EventContext]:
    """`jes` on the `pt` collection and `jer` on `eta`, both identifier-tagged `up` / `down`."""
    session, ctx = plain_context()
    pt = ctx["pt"]
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {"up": pt * JES_UP, "down": pt * JES_DOWN}}
    )
    eta = shifted["eta"]
    return session, graphed.vary(
        shifted, "jer", collections={"eta": {"up": eta * 1.01, "down": eta * 0.99}}
    )


def descendant_weight(ctx: EventContext) -> Any:
    """A weight read through a DESCENDANT context — the m48 row-space refusal, which `vary` raises
    AFTER `gather_members` has minted the call's labels."""
    mask = graphed.vary(ctx["pt"] > NOMINAL_CUT, "cut", lo=(ctx["pt"] > LOOSE_CUT))
    return ctx[mask]["pt"] * 0.5


# ---- R2 / §4.11-4: numerically tagged families ---------------------------------------------------
def numeric_families() -> tuple[Session, EventContext]:
    """Three families tagged `1` / `-1`: `jes` as a shift, `btag` and `jer` as weights. These are
    the families R2's literal `{"jes": 1, "btag": -1}` must reach, and every one of them moves the
    ambient weight by a different factor so no two universes materialize alike."""
    session, ctx = plain_context()
    pt = ctx["pt"]
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {"1": pt * JES_UP, "-1": pt * JES_DOWN}}
    )
    weight = shifted["pt"] * 0.5
    tagged = graphed.vary(
        shifted,
        "btag",
        weight,
        is_weight=True,
        variations={"1": weight * BTAG_UP, "-1": weight * BTAG_DOWN},
    )
    ambient = graphed.weight(tagged)
    return session, graphed.vary(
        tagged,
        "jer",
        ambient,
        is_weight=True,
        variations={"1": ambient * MU_UP, "-1": ambient * MU_DOWN},
    )


def dual_registered_family(*, numeric: bool = False) -> tuple[Session, EventContext]:
    """One family name registered as BOTH a shift and a weight — §4.8's mechanism for R1-b. Only
    the `up` tag is dual; `down` stays shift-only, which is what separates a per-(name, tag) verdict
    from a per-family one."""
    up, down = ("1", "-1") if numeric else ("up", "down")
    session, ctx = plain_context()
    pt = ctx["pt"]
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {up: pt * JES_UP, down: pt * JES_DOWN}}
    )
    weight = shifted["pt"] * 0.5
    return session, graphed.vary(
        shifted, "jes", weight, is_weight=True, variations={up: weight * 1.3}
    )


def identifier_families() -> tuple[Session, EventContext]:
    """The same programme spelled with IDENTIFIER tags — what §4.11-4 must refuse a number against."""
    session, ctx = plain_context()
    pt = ctx["pt"]
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {"up": pt * JES_UP, "down": pt * JES_DOWN}}
    )
    weight = shifted["pt"] * 0.5
    return session, graphed.vary(
        shifted,
        "btag",
        weight,
        is_weight=True,
        variations={"up": weight * BTAG_UP, "down": weight * BTAG_DOWN},
    )


# ---- §4.11-4's three context carriers, each in isolation -----------------------------------------
#: each carrier context registers TWO families on ONE carrier, so a legal two-coordinate point can
#: be written whose every axis is reachable through that carrier alone
CARRIERS = ("ambient_only_context", "collection_only_context", "selection_only_context")

#: `{carrier builder: the point whose axes only that carrier can supply}`
CARRIER_POINTS: dict[str, dict[str, str]] = {
    "ambient_only_context": {"jec": "up", "jes": "up"},
    "collection_only_context": {"jer": "up", "jer2": "up"},
    "selection_only_context": {"cut": "lo", "cut2": "hi"},
}


def ambient_only_context() -> tuple[Session, EventContext]:
    """`jes` and `jec` reachable through the ambient weight ALONE: the weight CARRIES their labels
    while its `_tags` is empty and no collection or selection mentions them. A `_tags`-derived walk
    refuses this context; the registry's points over the carrier's labels admit it (§8-g)."""
    session, record = source()
    pt = record["pt"]
    base = pt * 0.5
    minted = graphed.vary(base, "jes", up=base * JES_UP, down=base * JES_DOWN)
    minted = graphed.vary(minted, "jec", up=base * 1.02, down=base * 0.98)
    ambient = rebuild({label: graphed.universe(minted, label) for label in graphed.labels(minted)})
    return session, EventContext(session, pt, collections={"pt": pt}, weight=ambient)


def collection_only_context() -> tuple[Session, EventContext]:
    """`jer` / `jer2` reachable through a `Varied` COLLECTION alone — no ambient weight, no
    selection."""
    session, record = source()
    pt = record["pt"]
    collection = graphed.vary(pt, "jer", up=pt * 1.2, down=pt * 0.8)
    collection = graphed.vary(collection, "jer2", up=pt * 1.02, down=pt * 0.98)
    return session, EventContext(session, pt, collections={"pt": collection})


def selection_only_context() -> tuple[Session, EventContext]:
    """`cut` / `cut2` reachable through the SELECTION alone — the context carries no collections, so
    a re-indexed collection cannot supply the families instead."""
    session, record = source()
    pt = record["pt"]
    root = EventContext(session, pt, collections={})
    mask = graphed.vary(pt > NOMINAL_CUT, "cut", lo=(pt > LOOSE_CUT))
    mask = graphed.vary(mask, "cut2", hi=(pt > NOMINAL_CUT * 1.5))
    return session, root[mask]


def carrier(name: str) -> tuple[Session, EventContext]:
    return globals()[name]()


def carrier_weight(ctx: EventContext) -> Any:
    """A weight factor for a carrier context. The selection-only carrier holds no collections, so
    its factor is read off the selection and re-indexed by `vary` (the m48 parent-read path)."""
    selection = graphed.selection(ctx)
    if selection is None:
        return ctx["pt"] * 0.5
    return graphed.nominal(selection) * 0.0 + 1.0


def materialized(session: Session, container: Varied) -> dict[str, tuple[float, ...]]:
    """Every universe of `container`, as plain tuples — the shape a value comparison wants."""
    return {
        label: tuple(float(v) for v in session.materialize(graphed.universe(container, label)))
        for label in graphed.labels(container)
    }
