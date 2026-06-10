"""M11: the backend-idiom factorization (dask.array parity P0.1/P0.3, re-factored by design review).

``graphed.Array`` carries ONLY what is common to the backend idioms — operators, ufunc dispatch,
field access, and protected infrastructure (``_form_meta``). Idiomatic surfaces (numpy's
method/property style, awkward's functions-only style) live in the backend packages: a backend
supplies its proxy subclass through the ``array_type`` factory and every Session builder returns
it. The base class must NEVER grow numpy-specific members.
"""

from __future__ import annotations

import numpy as np
import pytest
from m11_toy import IdiomArray, IdiomBackend, MetaForm, ToyBackend, ToyForm, meta_source, source

from graphed import Array, Session


def test_backends_supply_their_proxy_class_through_every_builder() -> None:
    s = Session(IdiomBackend())
    a = s.source("a", form=ToyForm("source"), data=None)
    assert type(a) is IdiomArray
    assert type(a + a) is IdiomArray  # record_op
    assert type(a.map(lambda x: x)) is IdiomArray  # record_external
    assert type(a.double()) is IdiomArray  # subclass surface composes with recording


def test_backends_without_a_factory_get_the_base_array() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert type(a) is Array


def test_base_array_has_no_numpy_idiomatic_surface() -> None:
    # the factorization pin: numpy's member-function/property idiom must NOT exist on the shared
    # proxy (awkward's design applies operations as functions over arrays, never methods)
    for name in ("sum", "mean", "std", "var", "prod", "argmin", "argmax", "shape", "dtype", "ndim"):
        assert name not in vars(Array), f"numpy-idiomatic {name!r} leaked onto graphed.Array"
    assert "__array_function__" not in vars(Array)


def test_base_array_attribute_access_still_records_fields() -> None:
    # M3 semantics on the base class: .shape/.sum are record-field access, nothing more
    s = Session(ToyBackend())
    a = source(s, "a")
    out = a.shape
    assert isinstance(out, Array)
    assert s.form(out).describe() == "field"


def test_form_meta_infrastructure_delegates_and_falls_back() -> None:
    # subclass surfaces are built from _form_meta: form carries the metadata -> answered with no
    # graph growth; bare form -> falls back to field recording
    s = Session(IdiomBackend())
    m = s.source("m", form=MetaForm("source", (None, 3), "float64", 2), data=None)
    n = s.node_count()
    assert m.shape == (None, 3)
    assert s.node_count() == n
    bare = s.source("b", form=ToyForm("source"), data=None)
    assert isinstance(bare.shape, Array)


def test_numpy_api_functions_do_not_dispatch_on_the_base_array() -> None:
    # without __array_function__ on the shared proxy, np.sum falls into numpy's method-dispatch
    # path and fails: the numpy calling idiom belongs to the backend's proxy, not the base class
    s = Session(ToyBackend())
    a = meta_source(s, "a")
    with pytest.raises(TypeError):
        np.sum(a)
