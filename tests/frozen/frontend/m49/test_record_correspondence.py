"""§8.2(i): the record→reduced correspondence `compile_ir` carries on the compiled artifact.

Worded over the ACCESSOR, never over `variation_labels` — that field's only bound producer is
`graphed-histogram`'s group-plan builder, and no bound producer exists in a `compile_ir` program.
"""

from __future__ import annotations

from typing import Any

from m49_vary_fixtures import CHAIN_OPS, FORK_FACTORS, Topology, shared_node_topology, vary_topology

import graphed
import graphed.core as core
from graphed import Session, compile_ir
from graphed.numpy import NumpyBackend


def _compile(session: Session, topology: Topology) -> tuple[Any, list[dict[str, Any]]]:
    compiled = compile_ir(session, *topology.outputs)
    return compiled, core.GraphStore.deserialize(compiled.ir).nodes()


def _stage_ids(nodes: list[dict[str, Any]]) -> set[int]:
    return {i for i, node in enumerate(nodes) if node["kind"] == "stage"}


def test_every_surviving_record_id_keys_a_node_of_the_reduced_store() -> None:
    session = Session(NumpyBackend())
    topology = vary_topology(session, dead=True)
    compiled, nodes = _compile(session, topology)
    node_map = compiled.correspondence.node_map
    node_ids = set(range(len(nodes)))

    assert node_map.get(topology.dead) is None
    for record_id in topology.every_recorded_id() - {topology.dead}:
        reduced_id, member_index = node_map[record_id]
        assert reduced_id in node_ids
        if member_index is None:
            continue
        assert nodes[reduced_id]["kind"] == "stage"
        assert 0 <= member_index < len(nodes[reduced_id]["members"])


def test_the_map_partitions_the_topology_onto_the_reduced_store() -> None:
    """Without this clause a degenerate constant map `record_id -> (one_stage_id, 0)` satisfies
    every other clause. Read off the compiled artifact, never as §3.3's raw-builder literals."""
    session = Session(NumpyBackend())
    topology = vary_topology(session, dead=True)
    compiled, nodes = _compile(session, topology)
    node_map = compiled.correspondence.node_map

    prefix_targets = {node_map[record_id][0] for record_id in topology.prefix}
    assert len(prefix_targets) == 1

    per_universe = {
        label: {node_map[record_id][0] for record_id in chain} for label, chain in topology.chains.items()
    }
    assert all(len(targets) == 1 for targets in per_universe.values())
    assert len({targets.pop() for targets in per_universe.values()}) == len(FORK_FACTORS)

    live = [r for r in topology.every_recorded_id() if node_map.get(r) is not None]
    assert {node_map[r][0] for r in live} == set(range(len(nodes)))
    assert {node_map[r][0] for r in live if node_map[r][1] is not None} == _stage_ids(nodes)


def test_two_labels_sharing_a_node_collapse_onto_one_key() -> None:
    """The SET-VALUED half of §8.2(i)'s key space, witnessed here as a key collapse: the shared
    record ids are recovered through §3.4's impact verb."""
    session = Session(NumpyBackend())
    topology = shared_node_topology(session)
    compiled, _nodes = _compile(session, topology)
    node_map = compiled.correspondence.node_map
    impact = graphed.impact_by_label([topology.total])

    def keys(label: str) -> set[tuple[int, int | None]]:
        return {node_map[record_id] for record_id in impact[label] if node_map.get(record_id) is not None}

    shared_key = node_map[topology.shared]
    assert shared_key in keys("jes_up")
    assert shared_key in keys("jes_down")
    assert keys("jes_up") & keys("jes_down") == {shared_key}


def test_an_identity_token_maps_to_the_node_its_input_landed_in() -> None:
    """The NARROW discriminator isolating the post-DCE passes: `x * 1.0` is reachable from an
    output so it survives DCE, and is then folded away by the engine's identity rule."""
    session = Session(NumpyBackend())
    topology = vary_topology(session, identity=True)
    compiled, nodes = _compile(session, topology)
    node_map = compiled.correspondence.node_map

    landed = node_map[topology.token_input][0]
    assert node_map[topology.token][0] == landed
    assert len(nodes[landed]["members"]) == CHAIN_OPS + 1


def test_the_incremental_path_answers_the_same_record_keyed_map() -> None:
    """`Session(incremental=True)` reduces through a canonical arena of its own before the four
    passes; an accessor built on the one-shot path alone mis-keys every incremental program."""
    one_shot = Session(NumpyBackend())
    one_shot_topology = vary_topology(one_shot, dead=True)
    one_shot_compiled, _ = _compile(one_shot, one_shot_topology)

    incremental = Session(NumpyBackend(), incremental=True)
    incremental_topology = vary_topology(incremental, dead=True)
    incremental_compiled, _ = _compile(incremental, incremental_topology)

    assert incremental_topology.every_recorded_id() == one_shot_topology.every_recorded_id()
    assert dict(incremental_compiled.correspondence.node_map) == dict(one_shot_compiled.correspondence.node_map)
