"""M12: axis/keepdims-aware reductions + scans on Array (dask.array parity P1.4).

The recording rule is structural, and it is THE design decision of this milestone: a reduction
over the partitioned axis (``axis=None`` or ``axis=0``) records a **boundary reduction node**
(stage-fusion boundary, executed by the M7 tree reduction), while a reduction over an inner axis
(``axis>=1``) is partition-local and records a **plain fusible op**. Scans (cumsum/cumprod) are
always recorded fusible.
"""

from __future__ import annotations

import numpy as np
import pytest
from m12_toy import ToyBackend, meta_source, recorded, source

from graphed import Session

KINDS = ["sum", "prod", "mean", "std", "var", "min", "max", "any", "all", "argmin", "argmax"]

NAN_FUNCS = [
    (np.nansum, "nansum"),
    (np.nanprod, "nanprod"),
    (np.nanmean, "nanmean"),
    (np.nanstd, "nanstd"),
    (np.nanvar, "nanvar"),
    (np.nanmin, "nanmin"),
    (np.nanmax, "nanmax"),
    (np.nanargmin, "nanargmin"),
    (np.nanargmax, "nanargmax"),
]


@pytest.mark.parametrize("kind", KINDS)
def test_global_method_records_a_boundary_reduction(kind: str) -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    node = recorded(s, getattr(a, kind)())
    assert node["kind"] == "reduction"
    assert node["name"] == kind
    assert node["params"] == {}


def test_axis_zero_is_a_boundary_reduction_with_axis_param() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    node = recorded(s, a.sum(axis=0))
    assert node["kind"] == "reduction"
    assert node["params"] == {"axis": 0}


def test_inner_axis_is_a_fusible_op() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    node = recorded(s, a.sum(axis=1))
    assert node["kind"] == "op"
    assert node["name"] == "sum"
    assert node["params"] == {"axis": 1}


def test_negative_axis_normalizes_against_the_form_ndim() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    assert a.sum(axis=-1).node_id == a.sum(axis=1).node_id
    assert a.sum(axis=-2).node_id == a.sum(axis=0).node_id


def test_negative_axis_without_form_ndim_raises() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")  # bare ToyForm: no ndim to normalize against
    with pytest.raises(TypeError):
        a.sum(axis=-1)


def test_keepdims_and_ddof_are_recorded_only_when_non_default() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert recorded(s, a.sum(keepdims=True))["params"] == {"keepdims": True}
    assert recorded(s, a.std(ddof=1))["params"] == {"ddof": 1}
    # defaults intern with the bare form: np.std(a, ddof=0) IS a.std()
    assert a.std(ddof=0).node_id == a.std().node_id
    assert np.std(a, ddof=0).node_id == a.std().node_id


def test_numpy_functions_intern_with_the_methods() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    assert np.mean(a).node_id == a.mean().node_id
    assert np.sum(a, 1).node_id == a.sum(axis=1).node_id  # positional axis
    assert np.argmax(a).node_id == a.argmax().node_id
    assert np.sum(a).node_id == a.reduce("sum").node_id  # the M11 pin still holds


@pytest.mark.parametrize(("fn", "name"), NAN_FUNCS, ids=[n for _, n in NAN_FUNCS])
def test_nan_variants_record_their_own_canonical_ops(fn: object, name: str) -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    node = recorded(s, fn(a))  # type: ignore[operator]
    assert node["kind"] == "reduction"
    assert node["name"] == name


def test_scans_record_fusible_ops() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    flat = recorded(s, np.cumsum(a))
    assert flat["kind"] == "op"
    assert flat["name"] == "cumsum"
    assert flat["params"] == {}
    axis = recorded(s, a.cumprod(axis=1))
    assert axis["kind"] == "op"
    assert axis["name"] == "cumprod"
    assert axis["params"] == {"axis": 1}
    assert np.nancumsum(a).node_id != np.cumsum(a).node_id  # nan-variant is its own op


def test_unsupported_reduction_kwargs_raise() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        np.sum(a, out=np.empty(1))
