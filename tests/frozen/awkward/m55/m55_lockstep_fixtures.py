"""Fixtures for the m55 lockstep-by-propagation suite (awkward backend).

The toy is one nuisance moving two collections. `Jet` is a jagged record; `MET` is `met_of` OF those
jets — the plan's stand-in for Type-1 MET — and is the context's own central member, which is what
makes a propagated container's nominal intern to the context's MET node (§2 item 3). `Muon` and
`RawMET` stay on the record, so the tree also owns a collection the context READS instead of
carrying.

The m55 spelling — a `Varied` as a collection value — is reached only inside the functions the tests
call, never at import, so the tree COLLECTS against a pre-m55 tree and fails at RUN time
(TEST_SANITY). The `m55_` prefix is load-bearing: the pytest `pythonpath` publishes cross-dir
helpers under a global name.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import awkward as ak
import numpy as np

import graphed
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward, gak
from graphed.context import EventContext
from graphed.errors import GraphedError

#: {(collection, label): node id} — the equivalence witness the accept legs compare
NodeMap = dict[tuple[str, str], int]

N_EVENTS = 24
SEED = 55

#: how much of the jets' pT sum the MET recoils against, so a jet shift MOVES the propagated MET
MET_COUPLING = 0.1
#: a cut on the raw MET that keeps roughly half the events, for the masked-child legs
MET_CUT = 120.0

JES = {"up": 1.05, "down": 0.95}
JER = {"up": 1.02, "down": 0.98}
MUSC = {"up": 1.01, "down": 0.99}
JES_TAGS = tuple(JES)
JER_TAGS = tuple(JER)
MUSC_TAGS = tuple(MUSC)

#: the tag the stacking legs add to `jes` once up/down are registered
FLAT = "flat"
FLAT_FACTOR = 1.01
#: the factor that moves a container's nominal OFF the context's central member (§2 item 3)
OFF_NOMINAL = 1.10
#: the unclustered-energy factors, a second family on MET alone
UNCLUSTERED = {"up": 1.03, "down": 0.97}


def _make_events() -> ak.Array:
    rng = np.random.default_rng(SEED)
    n_jets = rng.integers(2, 6, size=N_EVENTS)
    n_muons = rng.integers(1, 3, size=N_EVENTS)
    jets, muons = int(n_jets.sum()), int(n_muons.sum())
    return ak.Array(
        {
            "Jet": ak.unflatten(
                ak.zip(
                    {
                        "pt": rng.uniform(20.0, 120.0, jets),
                        "eta": rng.uniform(-2.4, 2.4, jets),
                        "phi": rng.uniform(-np.pi, np.pi, jets),
                    }
                ),
                n_jets,
            ),
            "Muon": ak.unflatten(
                ak.zip(
                    {
                        "pt": rng.uniform(5.0, 60.0, muons),
                        "eta": rng.uniform(-2.4, 2.4, muons),
                        "phi": rng.uniform(-np.pi, np.pi, muons),
                    }
                ),
                n_muons,
            ),
            "RawMET": ak.zip(
                {
                    "pt": rng.uniform(80.0, 200.0, N_EVENTS),
                    "phi": rng.uniform(-np.pi, np.pi, N_EVENTS),
                }
            ),
        }
    )


#: one synthetic dataset for the whole tree
EVENTS = _make_events()


def met_of(jets: Any, raw_met: Any) -> Any:
    """The toy Type-1 MET: the raw MET recoiling against the jets' pT sum, phi untouched.

    A pure function of `(jets, raw MET)`, so `met_of(varied_jets, raw)` IS the propagated MET and its
    nominal universe interns to the node the context carries.
    """
    return gak.with_field(raw_met, raw_met.pt - MET_COUPLING * gak.sum(jets.pt, axis=1), "pt")


def rescale(record: Any, factor: float) -> Any:
    """A record whose `pt` is scaled — the toy's one shift, for jets, MET and muons alike."""
    return gak.with_field(record, record.pt * factor, "pt")


@dataclass
class Toy:
    session: Session
    ctx: Any


def base_toy() -> Toy:
    """A fresh Session and the base context: `Jet` and the PROPAGATED `MET` as collections."""
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", EVENTS)
    ctx = EventContext(
        session,
        events,
        collections={"Jet": events.Jet, "MET": met_of(events.Jet, events.RawMET)},
    )
    return Toy(session, ctx)


# ---- the two spellings of one `collections=` map -------------------------------------------
def unpack(container: Any, name: str, tags: tuple[str, ...]) -> dict[str, Any]:
    """The hand form m55's container form must be equivalent to (§2 item 4)."""
    return {tag: graphed.member_of(container, f"{name}_{tag}") for tag in tags}


def spelled(
    spell: str, containers: Mapping[str, Any], name: str, tags: tuple[str, ...]
) -> dict[str, Any]:
    """One `collections=` map in either spelling: `"varied"` hands over the containers themselves
    (the m55 form), `"hand"` unpacks each into `{tag: record}`."""
    if spell == "hand":
        return {key: unpack(value, name, tags) for key, value in containers.items()}
    return dict(containers)


# ---- containers ----------------------------------------------------------------------------
def jes_containers(ctx: Any) -> dict[str, Any]:
    """The loose form on the context's CENTRAL jets, plus the MET propagated from it."""
    jets = ctx.Jet
    varied = graphed.vary(jets, "jes", **{tag: rescale(jets, f) for tag, f in JES.items()})
    return {"Jet": varied, "MET": met_of(varied, ctx.RawMET)}


def jer_containers(ctx: Any, nominal_jets: Any = None) -> dict[str, Any]:
    """A SECOND family on the context's central jets and the MET propagated from it — the container
    item 2 admits (it carries `jer` alone) and item 3 admits (its nominal is the context's)."""
    jets = graphed.nominal(ctx["Jet"]) if nominal_jets is None else nominal_jets
    varied = graphed.vary(jets, "jer", **{tag: rescale(jets, f) for tag, f in JER.items()})
    return {"Jet": varied, "MET": met_of(varied, ctx.RawMET)}


def muon_containers(ctx: Any) -> dict[str, Any]:
    """A family on `Muon`, which the context READS off the record rather than carrying."""
    muons = ctx.Muon
    return {
        "Muon": graphed.vary(muons, "musc", **{tag: rescale(muons, f) for tag, f in MUSC.items()})
    }


def inherited_container(ctx: Any) -> Any:
    """A `jer` container built ON the jes-varied collection: it inherits `jes`'s labels, so it
    carries two families (§2 item 2's first refusal)."""
    jets = graphed.nominal(ctx["Jet"])
    return graphed.vary(ctx["Jet"], "jer", **{tag: rescale(jets, f) for tag, f in JER.items()})


def joint_container(ctx: Any) -> Any:
    """A `jer` container whose MEMBERS read the jes-varied collection: the loose form fans out and
    mints the joint labels for the cross terms (§2 item 2's second refusal)."""
    jets = graphed.nominal(ctx["Jet"])
    return graphed.vary(jets, "jer", **{tag: rescale(ctx["Jet"], f) for tag, f in JER.items()})


def other_family_container(ctx: Any) -> Any:
    """A container varying `jer`, offered to a call registering `jes` (§2 item 2's third refusal)."""
    jets = ctx.Jet
    return graphed.vary(jets, "jer", **{tag: rescale(jets, f) for tag, f in JER.items()})


def rescaled_nominal_container(ctx: Any) -> Any:
    """A `jer` container built on RESCALED jets, so its nominal is not the context's central member
    (§2 item 3's refusal)."""
    jets = rescale(graphed.nominal(ctx["Jet"]), OFF_NOMINAL)
    return graphed.vary(jets, "jer", **{tag: rescale(jets, f) for tag, f in JER.items()})


def restacked_containers(ctx: Any) -> dict[str, Any]:
    """A `jes` container built ON the already-varied collection, adding `flat`: its labels are
    exactly `jes`'s, so item 2 admits it and `check_family` refuses the re-offered tags."""
    varied = graphed.vary(ctx["Jet"], "jes", **{FLAT: rescale(graphed.nominal(ctx["Jet"]), FLAT_FACTOR)})
    return {"Jet": varied, "MET": met_of(varied, ctx.RawMET)}


# ---- contexts ------------------------------------------------------------------------------
def jes_context(toy: Toy) -> Any:
    """`jes` registered by the HAND form, so every fixture that builds ON it is m55-free."""
    containers = jes_containers(toy.ctx)
    return graphed.vary(toy.ctx, "jes", collections=spelled("hand", containers, "jes", JES_TAGS))


def event_mask(ctx: Any) -> Any:
    """An unvaried per-event mask, so a masked child's row space is one the whole tree shares."""
    return ctx.RawMET.pt > MET_CUT


def unclustered_child(ctx: Any) -> Any:
    """A `vary`-link descendant: a second family on MET alone, built off the central member."""
    met = graphed.nominal(ctx["MET"])
    return graphed.vary(
        ctx,
        "unclustered",
        collections={"MET": {tag: rescale(met, f) for tag, f in UNCLUSTERED.items()}},
    )


# ---- programs, each in its own fresh Session ------------------------------------------------
def headline_program(spell: str) -> tuple[Toy, Any]:
    """A `Varied` jets member and the `Varied` MET propagated from it, registered as one family."""
    toy = base_toy()
    containers = jes_containers(toy.ctx)
    return toy, graphed.vary(toy.ctx, "jes", collections=spelled(spell, containers, "jes", JES_TAGS))


def mixed_collections(containers: Mapping[str, Any], spell: str) -> dict[str, Any]:
    """A `{tag: record}` Jet beside a MET in the spelling under test — the two shapes in one call."""
    collections = spelled("hand", containers, "jes", JES_TAGS)
    if spell == "varied":
        collections["MET"] = containers["MET"]
    return collections


def mixed_program(spell: str) -> tuple[Toy, Any]:
    """A `{tag: record}` Jet beside a `Varied` MET, registered as one family."""
    toy = base_toy()
    containers = jes_containers(toy.ctx)
    return toy, graphed.vary(toy.ctx, "jes", collections=mixed_collections(containers, spell))


def second_family_program(spell: str) -> tuple[Toy, Any]:
    """A second family built on the context's nominal jets — item 2's paired accept."""
    toy = base_toy()
    ctx = jes_context(toy)
    containers = jer_containers(ctx)
    return toy, graphed.vary(ctx, "jer", collections=spelled(spell, containers, "jer", JER_TAGS))


def stacking_program(spell: str) -> tuple[Toy, Any]:
    """A new tag on an already-registered family, built on the nominal, with MET propagated."""
    toy = base_toy()
    ctx = jes_context(toy)
    jets = graphed.nominal(ctx["Jet"])
    varied = graphed.vary(jets, "jes", **{FLAT: rescale(jets, FLAT_FACTOR)})
    containers = {"Jet": varied, "MET": met_of(varied, ctx.RawMET)}
    return toy, graphed.vary(ctx, "jes", collections=spelled(spell, containers, "jes", (FLAT,)))


def masked_child_program(spell: str) -> tuple[Toy, Any]:
    """A container built at the parent, registered at a masked child."""
    toy = base_toy()
    ctx = jes_context(toy)
    containers = jer_containers(ctx)
    child = ctx[event_mask(ctx)]
    return toy, graphed.vary(child, "jer", collections=spelled(spell, containers, "jer", JER_TAGS))


def vary_descendant_program(spell: str) -> tuple[Toy, Any]:
    """A container built at the parent, registered at a `vary`-link descendant."""
    toy = base_toy()
    ctx = jes_context(toy)
    containers = jer_containers(ctx)
    child = unclustered_child(ctx)
    return toy, graphed.vary(child, "jer", collections=spelled(spell, containers, "jer", JER_TAGS))


def record_read_program(spell: str) -> tuple[Toy, Any]:
    """A collection the context READS off the record, registered at the context it was built at."""
    toy = base_toy()
    ctx = jes_context(toy)
    containers = muon_containers(ctx)
    return toy, graphed.vary(ctx, "musc", collections=spelled(spell, containers, "musc", MUSC_TAGS))


# ---- witnesses -----------------------------------------------------------------------------
def nodes(ctx: Any, *collections: str) -> NodeMap:
    """`{(collection, label): node id}` over every label each named collection carries."""
    return {
        (name, label): graphed.member_of(ctx[name], label).node_id
        for name in collections
        for label in graphed.labels(ctx[name])
    }


def equivalence(
    build: Callable[[str], tuple[Toy, Any]], *collections: str
) -> tuple[NodeMap, NodeMap]:
    """One program in two fresh Sessions — `Varied` members, then hand-unpacked. Node ids are
    per-Session and a deterministic function of recording order, so the two maps compare."""
    varied = nodes(build("varied")[1], *collections)
    hand = nodes(build("hand")[1], *collections)
    return varied, hand


def values(toy: Toy, array: Any) -> Any:
    """A materialized array as nested lists — exact for both idioms."""
    return ak.to_list(toy.session.materialize(array))


def refused(toy: Toy, call: Callable[[], Any]) -> str:
    """The message of the `GraphedError` `call` must raise, with §2 item 5's transactional clause
    checked: the Session's point registry and its inverse are exactly what they were."""
    before = (dict(toy.session._points), dict(toy.session._points_by_point))
    try:
        call()
    except GraphedError as error:
        assert (dict(toy.session._points), dict(toy.session._points_by_point)) == before, (
            "a refused call bound points into the registry"
        )
        return str(error)
    raise AssertionError("the call was accepted; the contract refuses it")


def cites_node(message: str, node_id: int) -> bool:
    """Whether `message` names that node id, not merely a digit run containing it."""
    return re.search(rf"(?<!\d){node_id}(?!\d)", message) is not None
