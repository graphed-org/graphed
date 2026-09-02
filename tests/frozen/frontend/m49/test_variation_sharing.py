"""§5.2a arena delta and §5.2c reduced-stage shape, both against an independently built oracle."""

from __future__ import annotations

from m49_vary_fixtures import (
    ARENA_CHAIN_OPS,
    FORK_FACTORS,
    arena_program,
    copied_topology,
    raw_arena_program,
    raw_topology,
    vary_topology,
)

import graphed.core as core
from graphed import Session, compile_ir
from graphed.numpy import NumpyBackend

VARIED = ("jes_up",)
ALL_LABELS = tuple(FORK_FACTORS)


def _reduced(session: Session, outputs: list[object]) -> list[dict[str, object]]:
    return core.GraphStore.deserialize(compile_ir(session, *outputs).ir).nodes()


def _stages(nodes: list[dict[str, object]]) -> list[int]:
    return [i for i, node in enumerate(nodes) if node["kind"] == "stage"]


def _members(nodes: list[dict[str, object]]) -> int:
    return sum(len(node.get("members") or ()) for node in nodes)


def _source_consumers(nodes: list[dict[str, object]]) -> list[int]:
    sources = {i for i, node in enumerate(nodes) if node["kind"] == "source"}
    return [i for i, node in enumerate(nodes) if sources & set(node["inputs"])]


def _arena_delta(session: Session, labels: tuple[str, ...]) -> int:
    before = session.node_count()
    arena_program(session, labels)
    return session.node_count() - before


def test_second_universe_adds_only_its_own_nodes() -> None:
    span = Session(NumpyBackend())
    arena_program(span, ())
    delta = _arena_delta(span, VARIED)

    oracle = Session(NumpyBackend())
    raw_arena_program(oracle, ())
    before = oracle.node_count()
    raw_arena_program(oracle, VARIED)
    expected = oracle.node_count() - before

    assert expected == ARENA_CHAIN_OPS + 2
    assert delta == expected


def test_a_per_universe_prefix_copy_is_visible_in_the_arena_delta() -> None:
    """The control leg: the delta above only means something because a copying build differs."""
    copied = Session(NumpyBackend())
    copied_topology(copied, ("nominal",), chain=ARENA_CHAIN_OPS)
    before = copied.node_count()
    copied_topology(copied, ("nominal", *VARIED), chain=ARENA_CHAIN_OPS)
    assert copied.node_count() - before > ARENA_CHAIN_OPS + 2


def test_shared_prefix_reduces_into_exactly_one_stage() -> None:
    session = Session(NumpyBackend())
    nodes = _reduced(session, vary_topology(session).outputs)
    consumers = _source_consumers(nodes)
    assert len(consumers) == 1
    assert nodes[consumers[0]]["kind"] == "stage"


def test_reduced_stage_count_equals_the_no_vary_oracle() -> None:
    session = Session(NumpyBackend())
    varied_nodes = _reduced(session, vary_topology(session).outputs)

    oracle = Session(NumpyBackend())
    oracle_nodes = _reduced(oracle, raw_topology(oracle, ALL_LABELS))

    assert len(_stages(varied_nodes)) == len(_stages(oracle_nodes))
    assert len(varied_nodes) == len(oracle_nodes)


def test_a_per_universe_prefix_copy_is_visible_in_the_reduced_shape() -> None:
    """The control leg for the stage-shape witness: a copied prefix reaches the source once per
    universe and carries the prefix work that many times."""
    shared_session = Session(NumpyBackend())
    shared_nodes = _reduced(shared_session, vary_topology(shared_session).outputs)

    copied = Session(NumpyBackend())
    copied_nodes = _reduced(copied, copied_topology(copied, ALL_LABELS))

    assert len(_source_consumers(shared_nodes)) == 1
    assert len(_source_consumers(copied_nodes)) == len(ALL_LABELS)
    assert _members(copied_nodes) > _members(shared_nodes)
