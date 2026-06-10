"""M11: Array exposes .shape/.dtype/.ndim, delegated to the backend form (dask.array parity P0.1).

When the backend's form carries the metadata, the properties answer WITHOUT recording any node;
when it does not (the M3 toy form), attribute access falls back to recording a field op exactly as
before — backends that model 'shape' as a record field are unaffected.
"""

from __future__ import annotations

from m11_toy import ToyBackend, meta_source, source

from graphed import Array, Session


def test_meta_properties_delegate_to_the_form_without_recording() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", shape=(None, 3), dtype="float64")
    n = s.node_count()
    assert a.shape == (None, 3)
    assert a.dtype == "float64"
    assert a.ndim == 2
    assert s.node_count() == n  # pure metadata: no graph growth


def test_formless_metadata_falls_back_to_field_recording() -> None:
    # the M3 semantics survive: a backend whose form has no array metadata still treats
    # .shape/.dtype/.ndim as record-field access (one recorded node each)
    s = Session(ToyBackend())
    a = source(s, "a")
    out = a.shape
    assert isinstance(out, Array)
    assert s.form(out).describe() == "field"
    assert isinstance(a.dtype, Array)
    assert isinstance(a.ndim, Array)


def test_meta_properties_after_recorded_ops() -> None:
    # ops produce forms via the backend; the properties read THAT node's form, not the source's
    s = Session(ToyBackend())
    a = meta_source(s, "a")
    out = a + a
    assert isinstance(out.shape, Array)  # ToyBackend op_form returns a bare ToyForm -> fallback
