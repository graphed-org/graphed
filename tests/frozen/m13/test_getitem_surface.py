"""M13: the common indexing surface on the neutral Array (dask.array parity P2.6).

Slices and integer indexing are COMMON to both backend idioms (numpy and awkward arrays both
support ``a[1:5]``, ``a[3]``), so they live on the base proxy. Both consume/restructure the
partitioned axis, so they record **boundary reduction nodes** (the M12 structural rule); the
params carry only the slice fields the user gave, so equal slices intern. Everything
numpy-flavored (tuple subscripts, manipulation methods) stays in graphed-numpy.
"""

from __future__ import annotations

import pytest
from m13_toy import ToyBackend, recorded, source

from graphed import Array, Session


def test_slice_records_a_boundary_node_with_present_only_params() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    node = recorded(s, a[1:5])
    assert node["kind"] == "reduction"
    assert node["name"] == "slice"
    assert node["params"] == {"start": 1, "stop": 5}
    assert recorded(s, a[::2])["params"] == {"step": 2}
    assert recorded(s, a[:7])["params"] == {"stop": 7}
    assert recorded(s, a[-5:])["params"] == {"start": -5}


def test_equal_slices_intern_to_one_node() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert a[1:5].node_id == a[1:5].node_id
    assert a[1:5].node_id != a[1:6].node_id


def test_integer_index_records_a_boundary_node() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    node = recorded(s, a[3])
    assert node["kind"] == "reduction"
    assert node["name"] == "index"
    assert node["params"] == {"i": 3}
    assert recorded(s, a[-1])["params"] == {"i": -1}


def test_mask_and_field_keys_are_unchanged() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    m = source(s, "m")
    assert recorded(s, a[m])["name"] == "getitem"
    assert recorded(s, a["f"])["name"] == "field"


def test_idiom_specific_keys_are_refused_on_the_base_proxy() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        _ = a[1.5]
    with pytest.raises(TypeError):
        _ = a[True]  # a bool is not an integer index
    with pytest.raises(TypeError):
        _ = a[:, 1]  # tuple subscripts are the numpy idiom: graphed-numpy's NumpyArray
    with pytest.raises(TypeError):
        _ = a["a":"b"]  # slice fields must be ints


def test_no_manipulation_methods_leak_onto_the_base_array() -> None:
    # the factorization pin extends to M13
    for name in ("reshape", "ravel", "squeeze", "transpose", "T", "take", "clip", "round", "astype"):
        assert name not in vars(Array), f"numpy-idiomatic {name!r} leaked onto graphed.Array"
