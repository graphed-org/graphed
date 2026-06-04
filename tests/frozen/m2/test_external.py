"""Opaque callables record External nodes (M2)."""

from __future__ import annotations

from backends import ListBackend, from_list

from graphed import Session


def test_map_records_external_node_and_applies_callable() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    n0 = s.node_count()
    doubled = a.map(lambda xs: [x * 2 for x in xs], name="double")
    assert s.node_count() == n0 + 1
    assert s.materialize(doubled) == [2, 4, 6]


def test_distinct_callables_record_distinct_nodes() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    m1 = a.map(lambda xs: xs, name="f1")
    m2 = a.map(lambda xs: xs, name="f2")
    assert m1.node_id != m2.node_id


def test_identical_callables_intern() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    m1 = a.map(lambda xs: xs, name="same")
    m2 = a.map(lambda xs: xs, name="same")
    assert m1.node_id == m2.node_id  # same name+input -> interned
