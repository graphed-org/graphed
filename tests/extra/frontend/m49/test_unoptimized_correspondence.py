"""§8.2(i)'s correspondence on the `opt_level=0` arm, which M6 binds to be 1:1.

Unoptimized, nothing is re-indexed, so the map is the identity over the whole record arena and a
frame still reaches every key. Discriminating against the optimized artifact of the SAME program,
whose keys carry a member index — an accessor that answered the identity there would be wrong.
"""

from __future__ import annotations

from backends import ListBackend, from_list

from graphed import Session
from graphed.execute import compile_ir, evaluate_ir


def _program() -> tuple[Session, object]:
    session = Session(ListBackend())
    x = from_list(session, "x", [1.0, 2.0])
    return session, (x * x) + x


def test_the_unoptimized_artifact_carries_the_identity_correspondence_and_still_evaluates() -> None:
    session, out = _program()
    compiled = compile_ir(session, out, optimize=False)

    nodes = session._store.node_count()
    assert compiled.correspondence.node_map == {n: (n, None) for n in range(nodes)}
    assert [key for key, _frame in compiled.correspondence.frames] == [(n, None) for n in range(nodes)]
    assert evaluate_ir(compiled, ListBackend(), {"x": [1.0, 2.0]}) == [[2.0, 6.0]]


def test_the_optimized_artifact_of_the_same_program_does_not_answer_the_identity() -> None:
    """The control: fusion re-keys, so the identity map is a real claim about `optimize=False`."""
    session, out = _program()
    optimized = compile_ir(session, out).correspondence.node_map

    assert optimized != {n: (n, None) for n in range(session._store.node_count())}
    assert any(member is not None for _nid, member in optimized.values())
