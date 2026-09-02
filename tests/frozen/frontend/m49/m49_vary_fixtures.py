"""Awkward-free fixtures for the m49 shift-path frontend suite.

ci.yml's REQUIRED free-threaded job collects `tests/frozen/frontend` WHOLE with only
`pytest hypothesis numpy` installed, so nothing reachable from this tree may import
`graphed.awkward` (or pyarrow/hist/pandas) — §10/m49's import ceiling. The module basename carries
the `m49_` prefix because the same process also collects `frontend/m48/vary_fixtures.py`, and under
prepend import mode a shared bare name binds to whichever sibling directory imported first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import graphed
from graphed import Array, Session
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_array, from_record

#: the 1-D payload every recording fixture reads
VECTOR = np.arange(1.0, 13.0)

#: §3.3's builder shape at frontend scale: D shared-prefix ops, then per universe one fork op, K
#: chain ops and exactly one terminating reduction.
PREFIX_OPS = 6
CHAIN_OPS = 4

#: §5.2a's chain length. The expected delta is `CHAIN + 2` (fork + chain + terminating reduction)
#: and travels with THIS builder, so the oracle re-derives it rather than the suite assuming it.
ARENA_CHAIN_OPS = 50

#: one non-identity fork factor per universe. None of them is 1.0: an identity fork would be folded
#: by the engine's identity rule and land in the PREFIX stage, which would blur §8.2(i)'s
#: per-universe partitioning clause. `test_record_correspondence` uses 1.0 deliberately, as its own
#: post-DCE discriminator.
FORK_FACTORS = {"nominal": 1.05, "jes_up": 1.15, "jes_down": 0.95}


def _tag(label: str) -> str:
    return label.split("_", 1)[1]


def _chain_step(value: object, step: int) -> object:
    return value * (1.0 + 0.01 * (step + 1))


@dataclass
class Topology:
    """Record-time node ids of a `vary`-built §3.3 topology, by role."""

    outputs: list[Array]
    total: graphed.Varied
    source: int
    prefix: list[int]
    forks: dict[str, int]
    chains: dict[str, list[int]]
    reductions: list[int]
    dead: int | None = None
    shared: int | None = None
    token: int | None = None
    token_input: int | None = None

    def every_recorded_id(self) -> set[int]:
        ids = {self.source, *self.prefix, *self.forks.values(), *self.reductions}
        for chain in self.chains.values():
            ids |= set(chain)
        for extra in (self.dead, self.shared, self.token):
            if extra is not None:
                ids.add(extra)
        return ids


def shared_prefix(session: Session, *, depth: int = PREFIX_OPS) -> tuple[Array, list[int]]:
    """The source plus `depth` chained ops every universe consumes."""
    x = from_array(session, "x", VECTOR)
    value, ids = x, []
    for step in range(depth):
        value = value + float(step + 1)
        ids.append(value.node_id)
    return value, ids


def vary_topology(
    session: Session,
    *,
    chain: int = CHAIN_OPS,
    dead: bool = False,
    identity: bool = False,
) -> Topology:
    """§3.3's topology built through the public `graphed.vary` surface, every reduction marked."""
    prefix_value, prefix_ids = shared_prefix(session)
    source = session.source_ids()[0]
    forks = {label: prefix_value * factor for label, factor in FORK_FACTORS.items()}
    varied = graphed.vary(
        forks["nominal"], "jes", **{_tag(label): forks[label] for label in FORK_FACTORS if label != "nominal"}
    )
    chains: dict[str, list[int]] = {label: [] for label in FORK_FACTORS}
    token = token_input = None
    value = varied
    for step in range(chain):
        value = _chain_step(value, step)
        for label in FORK_FACTORS:
            chains[label].append(graphed.universe(value, label).node_id)
    if identity:
        token_input = graphed.universe(value, "nominal").node_id
        value = value * 1.0
        token = graphed.universe(value, "nominal").node_id
    total = value.sum()
    outputs = [graphed.universe(total, label) for label in graphed.labels(total)]
    topology = Topology(
        outputs=outputs,
        total=total,
        source=source,
        prefix=prefix_ids,
        forks={label: forks[label].node_id for label in FORK_FACTORS},
        chains=chains,
        reductions=[out.node_id for out in outputs],
        token=token,
        token_input=token_input,
    )
    if dead:
        topology.dead = (prefix_value * 99.0).node_id
    return topology


def shared_node_topology(session: Session, *, chain: int = CHAIN_OPS) -> Topology:
    """§3.4's shape: a derived node UPSTREAM of the fork that both varied members consume and the
    nominal member does not. Interning keys on input ids, so no node downstream of the fork can be
    shared by two labels carrying distinct members — the sharing has to sit above it."""
    prefix_value, prefix_ids = shared_prefix(session)
    source = session.source_ids()[0]
    shared = prefix_value * 2.0
    forks = {
        "nominal": prefix_value * FORK_FACTORS["nominal"],
        "jes_up": prefix_value * FORK_FACTORS["jes_up"] + shared,
        "jes_down": prefix_value * FORK_FACTORS["jes_down"] - shared,
    }
    varied = graphed.vary(forks["nominal"], "jes", up=forks["jes_up"], down=forks["jes_down"])
    chains: dict[str, list[int]] = {label: [] for label in FORK_FACTORS}
    value = varied
    for step in range(chain):
        value = _chain_step(value, step)
        for label in FORK_FACTORS:
            chains[label].append(graphed.universe(value, label).node_id)
    total = value.sum()
    outputs = [graphed.universe(total, label) for label in graphed.labels(total)]
    return Topology(
        outputs=outputs,
        total=total,
        source=source,
        prefix=prefix_ids,
        forks={label: forks[label].node_id for label in FORK_FACTORS},
        chains=chains,
        reductions=[out.node_id for out in outputs],
        shared=shared.node_id,
    )


def raw_topology(session: Session, labels: tuple[str, ...], *, chain: int = CHAIN_OPS) -> list[Array]:
    """The ORACLE for §5.2: the same topology hand-built WITHOUT `vary`, in its own Session."""
    prefix_value, _ = shared_prefix(session)
    outputs = []
    for label in labels:
        value = prefix_value * FORK_FACTORS[label]
        for step in range(chain):
            value = _chain_step(value, step)
        outputs.append(value.sum())
    return outputs


def copied_topology(session: Session, labels: tuple[str, ...], *, chain: int = CHAIN_OPS) -> list[Array]:
    """The COUNTERFACTUAL §5.2 exists to exclude: a per-universe COPY of the shared prefix. Each
    universe's prefix constants differ, so nothing interns and no prefix work is shared."""
    x = from_array(session, "x", VECTOR)
    outputs = []
    for universe, label in enumerate(labels):
        value = x
        for step in range(PREFIX_OPS):
            value = value + float(step + 1 + universe * 100)
        value = value * FORK_FACTORS[label]
        for step in range(chain):
            value = _chain_step(value, step)
        outputs.append(value.sum())
    return outputs


def arena_program(session: Session, labels: tuple[str, ...]) -> list[Array]:
    """§5.2a's SPAN builder at `ARENA_CHAIN_OPS` chain length. `labels` names the NON-nominal
    universes; the N=1 program has none, so it is `vary`-free by construction. Every universe ends
    in a terminating reduction — without it the expected delta would be `CHAIN + 1`."""
    prefix_value, _ = shared_prefix(session)
    value = prefix_value * FORK_FACTORS["nominal"]
    if labels:
        value = graphed.vary(
            value, "jes", **{_tag(label): prefix_value * FORK_FACTORS[label] for label in labels}
        )
    for step in range(ARENA_CHAIN_OPS):
        value = _chain_step(value, step)
    total = value.sum()
    if not labels:
        return [total]
    return [graphed.universe(total, label) for label in graphed.labels(total)]


def raw_arena_program(session: Session, labels: tuple[str, ...]) -> list[Array]:
    """§5.2a's ORACLE: the same universes hand-built WITHOUT `vary`. Its own node-count delta is
    where the expected integer comes from."""
    return raw_topology(session, ("nominal", *labels), chain=ARENA_CHAIN_OPS)


def reachable(session: Session, array: Array) -> set[int]:
    """Every record-time node id reachable from `array`, via the generic `session.walk`."""
    seen: set[int] = set()

    def note(node_id: int, *_rest: object) -> None:
        seen.add(node_id)

    session.walk(array, source=note, op=note, external=note)
    return seen


# ---- §5.3: the FLAT (branch-per-column) projection fixture ------------------------------------
@dataclass
class FlatProgram:
    session: Session
    varied: graphed.Varied
    nominal: Array
    shifted: Array
    source_nid: int


def flat_projection_program() -> FlatProgram:
    """A flat record whose shift reads one extra TOP-LEVEL column, plus a whole-record consumer.

    `read_columns` reports only fields read directly off the source node, so the extra column has to
    be a sibling branch (`Jet_eta`), never a nested field. The conservative member is spelled
    `ev.map(f)` — the elementwise spelling is ill-typed against the shipping backends on a record.
    """
    session = Session(NumpyBackend())
    ev = from_record(
        session,
        "ev",
        Jet_pt=np.arange(1.0, 5.0),
        Jet_eta=np.arange(0.0, 4.0),
        Muon_pt=np.arange(2.0, 6.0),
    )
    pt = ev["Jet_pt"]
    shifted = pt * (1.0 + 0.1 * ev["Jet_eta"])
    opaque = ev.map(lambda record: record["Jet_pt"] * 1.02)
    varied = graphed.vary(pt, "jes", up=shifted, opaque=opaque)
    return FlatProgram(
        session=session,
        varied=varied,
        nominal=pt,
        shifted=shifted,
        source_nid=session.source_ids()[0],
    )


# ---- §5.4: the self-contained, awkward-free join fixture ---------------------------------------
LEFT_KEY = np.array([1, 2, 3, 4])
LEFT_PT = np.array([10.0, 20.0, 30.0, 40.0])
RIGHT_KEY = np.array([2, 3, 5])
RIGHT_SF = np.array([1.5, 2.5, 3.5])

#: the relational reference, computed with numpy alone: an inner join on `key`.
JOINED_PT = np.array([20.0, 30.0])
JOINED_SF = np.array([1.5, 2.5])


@dataclass
class JoinProgram:
    session: Session
    left: Array
    right: Array
    joined: Array


def join_program() -> JoinProgram:
    """Two flat `from_record` tables joined through `graphed.join` on a `NumpyBackend`.

    Deliberately NOT `frontend/m40/shuffle_backends.py`: that module imports awkward, pandas and
    `graphed_corpus` at module scope, so it cannot be imported at all on the required
    free-threaded gate, and its directory is not on `pythonpath`.
    """
    session = Session(NumpyBackend())
    left = from_record(session, "L", key=LEFT_KEY, pt=LEFT_PT)
    right = from_record(session, "R", key=RIGHT_KEY, sf=RIGHT_SF)
    return JoinProgram(session=session, left=left, right=right, joined=graphed.join(left, right, on=["key"]))


# ---- the §2.6 context program the `_align` path needs ------------------------------------------
@dataclass
class ContextProgram:
    session: Session
    root: EventContext
    selected: EventContext
    varied: graphed.Varied
    mask_labels: tuple[str, ...] = field(default_factory=tuple)


NOMINAL_CUT = 4.0
LOOSE_CUT = 2.0
JES_SCALE = 1.1
BTAG_SCALE = 1.2


def varied_mask_context_program() -> ContextProgram:
    """A `vary` whose INHERITED members were read through the parent context while the container's
    handle is the mask-derived child — the one shape `vary._align` acts on. The mask is itself
    `Varied`, so each inherited member is re-indexed label-aligned per §2.4.
    """
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR)
    pt = record["pt"]
    root = EventContext(session, pt, collections={"pt": pt})
    quantity = root["pt"]
    varied_quantity = graphed.vary(quantity, "jes", up=quantity * JES_SCALE)
    mask = graphed.vary(quantity > NOMINAL_CUT, "cut", lo=(quantity > LOOSE_CUT))
    selected = root[mask]
    inside = selected["pt"]
    return ContextProgram(
        session=session,
        root=root,
        selected=selected,
        varied=graphed.vary(varied_quantity, "btag", up=inside * BTAG_SCALE),
        mask_labels=graphed.labels(mask),
    )
