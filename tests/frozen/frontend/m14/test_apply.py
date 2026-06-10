"""M14: ``graphed.apply`` — the multi-input blockwise external (dask.array parity P3.8).

`Array.map(fn)` (M2) wraps ONE array; dask's blockwise/map_blocks zips several. The common
infrastructure is a function over arrays (idiom-neutral — awkward style): `apply(fn, *arrays)`
records ONE External node with N inputs, carrying the backend's PayloadDescriptor so the opaque
callable stays a flagged preservation risk (plan A.3.1). Single-input `apply` IS `map` (interns).
"""

from __future__ import annotations

import pytest
from m14_toy import ToyBackend, recorded, source

from graphed import Array, Session, apply


def test_apply_records_one_external_with_n_inputs() -> None:
    s = Session(ToyBackend())
    a = source(s, "a", 3)
    b = source(s, "b", 4)
    out = apply(lambda x, y: x + y, a, b, name="add2")
    node = recorded(s, out)
    assert node["kind"] == "external"
    assert node["inputs"] == [a.node_id, b.node_id]
    descriptor = node["descriptor"]
    assert descriptor["kind"] == "opaque_callable"  # type: ignore[index]
    assert "add2" in descriptor["content_hash"]  # type: ignore[index]


def test_apply_materializes_by_calling_fn_on_all_inputs() -> None:
    s = Session(ToyBackend())
    a = source(s, "a", 3)
    b = source(s, "b", 4)
    c = source(s, "c", 5)
    out = apply(lambda x, y, z: x * y + z, a, b, c, name="fma")
    assert s.materialize(out) == 17


def test_single_input_apply_interns_with_map() -> None:
    def double(x: object) -> object:
        return x

    s = Session(ToyBackend())
    a = source(s, "a", 3)
    assert apply(double, a).node_id == a.map(double).node_id


def test_apply_needs_at_least_one_array_and_one_session() -> None:
    s1 = Session(ToyBackend())
    s2 = Session(ToyBackend())
    a = source(s1, "a", 1)
    b = source(s2, "b", 2)
    with pytest.raises(TypeError):
        apply(lambda: 0)
    with pytest.raises(TypeError):
        apply(lambda x, y: x, a, b)  # arrays from different sessions
    with pytest.raises(TypeError):
        apply(lambda x, y: x, a, 7)  # type: ignore[arg-type]


def test_the_gufunc_idiom_stays_out_of_the_base_array() -> None:
    # apply is a FUNCTION over arrays (idiom-neutral); the signature-aware apply_gufunc is
    # numpy-specific and lives in graphed-numpy
    assert "apply_gufunc" not in vars(Array)
    assert "blockwise" not in vars(Array)
