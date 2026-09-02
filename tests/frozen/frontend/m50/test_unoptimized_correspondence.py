"""§8.2(i) on the `opt_level=0` arm M6 binds to be 1:1.

Unoptimized, nothing is re-indexed: the correspondence is the identity over the whole record arena
and a frame reaches every key in record ids. Discriminating against the OPTIMIZED artifact of the
SAME program, whose fusion re-keys and stamps a member index — an accessor that answered the identity
there would be wrong, so the identity claim is a real property of `optimize=False`, not a tautology.

Awkward-free (a list backend + core only): safe under the required 3.14t free-threaded gate.
"""

from __future__ import annotations

from backends import ListBackend, from_list

from graphed import Session
from graphed.execute import compile_ir, evaluate_ir


def _program() -> tuple[Session, object]:
    session = Session(ListBackend())
    x = from_list(session, "x", [1.0, 2.0])
    return session, (x * x) + x


def test_the_unoptimized_artifact_is_the_identity_over_record_ids_and_evaluates() -> None:
    session, out = _program()
    compiled = compile_ir(session, out, optimize=False)
    nodes = session._store.node_count()

    identity = {n: (n, None) for n in range(nodes)}
    assert dict(compiled.correspondence.node_map) == identity
    assert [key for key, _frame in compiled.correspondence.frames] == [(n, None) for n in range(nodes)]
    assert evaluate_ir(compiled, ListBackend(), {"x": [1.0, 2.0]}) == [[2.0, 6.0]]


def test_the_optimized_artifact_of_the_same_program_does_not_answer_the_identity() -> None:
    """The control that makes the identity a claim about `optimize=False`: fusion re-keys the same
    program and stamps a member index, so a member index is present exactly where the unoptimized
    arm carries None."""
    session, out = _program()
    optimized = dict(compile_ir(session, out).correspondence.node_map)

    assert optimized != {n: (n, None) for n in range(session._store.node_count())}
    assert any(member is not None for _nid, member in optimized.values())
