"""§4.1: one correctionlib payload, N parameterizations — the payload is never duplicated.

The canonical weight-variation form varies the existing `systematic` CATEGORY parameter of ONE
payload, so all labels' `External` nodes share a single `PayloadDescriptor.content_hash`
(§A.3.1 reproducibility). `gak.apply_correction` is NOT that path: it records
`params={"name": ..., "args": json.dumps(args)}`, so the systematic value rides inside a JSON
string and this observable is unsatisfiable through it.

Fill-free, yet it lives in the awkward tree: `record_external` yields a payload descriptor only
under `AwkwardBackend` — `NumpyBackend` raises "backend returned no payload descriptor for
external op". `correctionlib` itself is never imported; the plugin loads lazily.
"""

from __future__ import annotations

import agc
from vary_ctx_fixtures import awkward_session

import graphed
from graphed.awkward import gak
from graphed.preserve import CORRECTIONLIB_PLUGIN, record_external

SYSTEMATICS = ("nominal", "up", "down")


def _varied_scale_factor() -> tuple[graphed.Session, graphed.Varied]:
    session, root = awkward_session()
    njet = gak.num(root.Jet)
    payload = agc.correctionlib_json()  # the m9 fixture at its default scale=1.0
    members = {
        systematic: record_external(
            session,
            CORRECTIONLIB_PLUGIN,
            payload,
            [njet],
            params={"name": "event_sf", "systematic": systematic},
        )
        for systematic in SYSTEMATICS
    }
    varied = graphed.vary(members["nominal"], "sf", up=members["up"], down=members["down"])
    return session, varied


def _external_nodes(session: graphed.Session, varied: graphed.Varied) -> dict[str, dict[str, object]]:
    nodes = session._store.nodes()
    return {label: nodes[graphed.universe(varied, label).node_id] for label in graphed.labels(varied)}


def test_all_labels_share_one_payload_content_hash() -> None:
    session, varied = _varied_scale_factor()
    nodes = _external_nodes(session, varied)
    assert len(nodes) == 3
    hashes = {node["descriptor"]["content_hash"] for node in nodes.values()}  # type: ignore[index]
    assert len(hashes) == 1, "the payload was duplicated per label; §A.3.1 forbids it"
    assert all(node["kind"] == "external" for node in nodes.values())


def test_the_labels_differ_only_in_the_systematic_param() -> None:
    session, varied = _varied_scale_factor()
    nodes = _external_nodes(session, varied)
    params = {label: dict(node["params"]) for label, node in nodes.items()}  # type: ignore[arg-type]
    assert {p.pop("systematic") for p in params.values()} == set(SYSTEMATICS)
    remaining = list(params.values())
    assert all(other == remaining[0] for other in remaining[1:])
    assert remaining[0]["name"] == "event_sf"


def test_the_three_universes_are_distinct_nodes_over_one_input() -> None:
    """Only a real content difference forks identity (§1.2): the `systematic=` param is one, so
    the nodes differ — while their single shared input proves nothing upstream was recomputed."""
    session, varied = _varied_scale_factor()
    nodes = _external_nodes(session, varied)
    assert len({graphed.universe(varied, label).node_id for label in graphed.labels(varied)}) == 3
    inputs = {tuple(node["inputs"]) for node in nodes.values()}  # type: ignore[arg-type]
    assert len(inputs) == 1
