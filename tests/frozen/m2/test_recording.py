"""Recording: one interned node + one form per op; repeated sub-expressions add zero nodes (M2)."""

from __future__ import annotations

from backends import ListBackend, from_list

from graphed import Session


def test_one_node_and_one_form_per_op() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [4, 5, 6])
    assert s.node_count() == 2
    c = a + b
    assert s.node_count() == 3  # exactly one new node
    assert s.form(c).describe() == "int"  # exactly one form


def test_repeated_subexpression_records_zero_new_nodes() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [4, 5, 6])
    first = a + b
    n_after_first = s.node_count()
    second = a + b  # identical structure -> interns to the same node
    assert second.node_id == first.node_id
    assert s.node_count() == n_after_first


def test_distinct_ops_distinct_nodes() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [4, 5, 6])
    add = a + b
    mul = a * b
    assert add.node_id != mul.node_id
    assert s.node_count() == 4  # a, b, add, mul


def test_reduction_records_a_node() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    total = a.reduce("sum")
    assert s.form(total).describe() == "scalar"
    assert s.node_count() == 2
