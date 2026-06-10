"""M12: the axis/keepdims-aware reduction + scan INFRASTRUCTURE (dask.array parity P1.4).

The recording rule is structural, and it is THE design decision of this milestone: a reduction
over the partitioned axis (``axis=None`` or ``axis=0``) records a **boundary reduction node**
(stage-fusion boundary, executed by the M7 tree reduction), while a reduction over an inner axis
(``axis>=1``) is partition-local and records a **plain fusible op**. Scans (cumsum/cumprod) are
always recorded fusible.

Per the M11 factorization these are protected infrastructure on the base Array (`_reduction`,
`_scan`, `_norm_axis`); the suite pins them through a minimal backend proxy subclass. The numpy
method/function idiom over the same infrastructure is pinned in graphed-numpy's M12 suite.
"""

from __future__ import annotations

import pytest
from m12_toy import ToyBackend, meta_source, recorded, source

from graphed import Array, Session

KINDS = ["sum", "prod", "mean", "std", "var", "min", "max", "any", "all", "argmin", "argmax"]
NAN_KINDS = ["nansum", "nanprod", "nanmean", "nanstd", "nanvar", "nanmin", "nanmax", "nanargmin", "nanargmax"]


@pytest.mark.parametrize("kind", KINDS + NAN_KINDS)
def test_global_reduction_records_a_boundary_reduction(kind: str) -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    node = recorded(s, a.red(kind))
    assert node["kind"] == "reduction"
    assert node["name"] == kind
    assert node["params"] == {}


def test_axis_zero_is_a_boundary_reduction_with_axis_param() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    node = recorded(s, a.red("sum", axis=0))
    assert node["kind"] == "reduction"
    assert node["params"] == {"axis": 0}


def test_inner_axis_is_a_fusible_op() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    node = recorded(s, a.red("sum", axis=1))
    assert node["kind"] == "op"
    assert node["name"] == "sum"
    assert node["params"] == {"axis": 1}


def test_negative_axis_normalizes_against_the_form_ndim() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    assert a.red("sum", axis=-1).node_id == a.red("sum", axis=1).node_id
    assert a.red("sum", axis=-2).node_id == a.red("sum", axis=0).node_id


def test_negative_axis_without_form_ndim_raises() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")  # bare ToyForm: no ndim to normalize against
    with pytest.raises(TypeError):
        a.red("sum", axis=-1)


def test_non_integer_axis_raises() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    with pytest.raises(TypeError):
        a.red("sum", axis=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        a.red("sum", axis=True)  # type: ignore[arg-type]


def test_keepdims_and_ddof_are_recorded_only_when_non_default() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert recorded(s, a.red("sum", keepdims=True))["params"] == {"keepdims": True}
    assert recorded(s, a.red("std", ddof=1))["params"] == {"ddof": 1}
    # defaults intern with the bare form: std(ddof=0) IS std()
    assert a.red("std", ddof=0).node_id == a.red("std").node_id


def test_infrastructure_interns_with_the_m2_reduce() -> None:
    s = Session(ToyBackend())
    a = source(s, "a")
    assert a.red("sum").node_id == a.reduce("sum").node_id  # the M2/M11 pin still holds


def test_scans_record_fusible_ops() -> None:
    s = Session(ToyBackend())
    a = meta_source(s, "a", ndim=2)
    flat = recorded(s, a.scan("cumsum"))
    assert flat["kind"] == "op"
    assert flat["name"] == "cumsum"
    assert flat["params"] == {}
    axis = recorded(s, a.scan("cumprod", axis=1))
    assert axis["kind"] == "op"
    assert axis["params"] == {"axis": 1}
    assert a.scan("cumsum", axis=-1).node_id == a.scan("cumsum", axis=1).node_id
    assert a.scan("nancumsum").node_id != a.scan("cumsum").node_id  # nan-variant is its own op


def test_no_reduction_methods_leak_onto_the_base_array() -> None:
    # the factorization pin extends to M12: the numpy method idiom stays in graphed-numpy
    for name in ("sum", "mean", "std", "var", "cumsum", "cumprod"):
        assert name not in vars(Array), f"numpy-idiomatic {name!r} leaked onto graphed.Array"
