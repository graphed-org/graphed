"""§8.2(i): the incremental path composes its own original->canonical map in front of the four passes.

The m49 incremental anchor rides `vary_topology(dead=True)`, whose record arena canonicalizes 1:1
(node_count == canonical_count), so the compose step is a no-op there and its deletion ships green.
This fixture threads an algebraic-identity node the engine folds, so the reducer's canonical arena is
strictly smaller than the record arena: the compose becomes load-bearing and the same record-keyed
equality the m49 anchor asserts now discriminates it.

Awkward-free by construction (core + numpy only): the required 3.14t free-threaded gate collects this
subtree with only `pytest hypothesis numpy` installed.
"""

from __future__ import annotations

import numpy as np

from graphed import Array, Session
from graphed.execute import compile_ir
from graphed.numpy import NumpyBackend, from_array

VALUES = np.arange(1.0, 5.0)


def _fold_program(session: Session) -> tuple[Array, Array, Array]:
    """A program whose middle node is `squared * 1.0` — folded onto `squared` by the engine's
    identity rule, so its record id maps to an EARLIER canonical id."""
    x = from_array(session, "x", VALUES)
    squared = x * x
    folded = squared * 1.0
    return (folded + x), squared, folded


def test_the_fixture_arena_canonicalizes_below_the_record_arena() -> None:
    """Non-vacuity guard: the anchor only discriminates while the original->canonical map is
    non-identity. If a future engine stopped folding here the arenas would coincide and this test
    would silently regress to the m49 anchor's vacuity — so pin the gap explicitly."""
    session = Session(NumpyBackend(), incremental=True)
    _fold_program(session)
    assert session._reducer.canonical_count() < session._store.node_count()


def test_the_incremental_path_answers_the_full_record_keyed_map() -> None:
    one_shot = Session(NumpyBackend())
    one_out, _, _ = _fold_program(one_shot)
    one_map = dict(compile_ir(one_shot, one_out).correspondence.node_map)

    incremental = Session(NumpyBackend(), incremental=True)
    inc_out, _, _ = _fold_program(incremental)
    inc_map = dict(compile_ir(incremental, inc_out).correspondence.node_map)

    # keys the WHOLE record arena, not the smaller canonical arena — the half a dropped compose
    # loses: it would answer in canonical ids, `canonical_count` keys short of the top record ids.
    assert set(inc_map) == set(range(one_shot._store.node_count()))
    assert inc_map == one_map


def test_a_folded_record_id_lands_on_its_input_key() -> None:
    """The folded node and its input share one reduced key — the record-keyed image of a
    non-identity original->canonical entry, still resolved through the compose."""
    session = Session(NumpyBackend(), incremental=True)
    _out, squared, folded = _fold_program(session)
    node_map = dict(compile_ir(session, _out).correspondence.node_map)
    assert node_map[folded.node_id] == node_map[squared.node_id]
