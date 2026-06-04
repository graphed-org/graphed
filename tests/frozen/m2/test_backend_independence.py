"""Backend-independence: same program, two backends -> identical structure, different results (M2)."""

from __future__ import annotations

from backends import ListBackend, NegListBackend, from_list

from graphed import Session


def _program(backend: object) -> tuple[Session, object]:
    s = Session(backend)  # type: ignore[arg-type]
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [10, 20, 30])
    c = (a + b) * a
    total = c.reduce("sum")
    return s, total


def test_recorded_structure_is_backend_independent() -> None:
    s1, _ = _program(ListBackend())
    s2, _ = _program(NegListBackend())
    assert s1.node_count() == s2.node_count()
    assert s1.to_dot() == s2.to_dot()  # byte-identical recorded graph


def test_results_depend_on_the_backend() -> None:
    s1, out1 = _program(ListBackend())
    s2, out2 = _program(NegListBackend())
    r1 = s1.materialize(out1)
    r2 = s2.materialize(out2)
    assert r1 != r2  # same graph, different evaluation
    # (a+b)*a summed = (11*1 + 22*2 + 33*3) = 154 ; negated backend -> -154
    assert r1 == 154
    assert r2 == -154
