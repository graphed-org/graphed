"""M17: list-of-strings getitem on the neutral proxy (dask-awkward parity P1).

``a[["x", "y"]]`` selects a RECORD SUBSET — a lazy column narrowing both array models share
(awkward record arrays and numpy structured arrays). It records ONE fusible op whose params carry
the field list canonically, so equal subsets intern.
"""

from __future__ import annotations

import pytest
from m17_toy import ToyBackend, recorded, source

from graphed import Session


def test_list_of_strings_records_one_fields_subset_op() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    out = a[["x", "y"]]
    node = recorded(s, out)
    assert node["kind"] == "op"
    assert node["name"] == "fields"
    assert node["params"] == {"fields": "x,y"}


def test_subsets_intern_and_order_matters() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert a[["x", "y"]].node_id == a[["x", "y"]].node_id
    assert a[["x", "y"]].node_id != a[["y", "x"]].node_id  # field order is part of the selection


def test_tuples_and_mixed_lists_are_still_refused() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        _ = a[["x", 3]]  # type: ignore[list-item]
    with pytest.raises(TypeError):
        _ = a[[]]  # an empty selection is meaningless
    with pytest.raises(TypeError):
        _ = a[("x", "y")]  # tuples stay the (refused) numpy idiom on the base proxy
