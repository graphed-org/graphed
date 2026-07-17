"""M12: axis/keepdims-aware reductions and scans evaluate exactly as numpy (parity P1.4).

Reference semantics are numpy's own: every case materializes the deferred reduction and compares
against the same call on the raw array. Forms are pinned too: a global reduction is a scalar,
``axis=0`` consumes the partitioned axis (concrete trailing shape), ``axis>=1`` preserves it.
"""

from __future__ import annotations

import numpy as np
import pytest
from graphed import GraphedTypeError, Session

from graphed_numpy import NumpyBackend, from_array

D2 = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0], [3.0, 5.0, 8.0]])
DN = np.array([[3.0, np.nan, 4.0], [np.nan, 5.0, 9.0], [2.0, 6.0, np.nan], [3.0, 5.0, 8.0]])
DB = np.array([[True, False, True], [False, False, True], [True, True, True], [False, False, False]])

KINDS = ["sum", "prod", "mean", "std", "var", "min", "max", "argmin", "argmax"]
NAN_FUNCS = [np.nansum, np.nanprod, np.nanmean, np.nanstd, np.nanvar, np.nanmin, np.nanmax]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_reduction_matches_numpy(kind: str, axis: int | None) -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", D2)
    out = getattr(a, kind)(axis=axis)
    np.testing.assert_array_equal(np.asarray(s.materialize(out)), getattr(np, kind)(D2, axis=axis))


@pytest.mark.parametrize("kind", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_keepdims_matches_numpy(kind: str, axis: int | None) -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", D2)
    out = getattr(a, kind)(axis=axis, keepdims=True)
    np.testing.assert_array_equal(
        np.asarray(s.materialize(out)), getattr(np, kind)(D2, axis=axis, keepdims=True)
    )


@pytest.mark.parametrize("fn", NAN_FUNCS, ids=[f.__name__ for f in NAN_FUNCS])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_nan_variants_match_numpy(fn: np.ufunc, axis: int | None) -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", DN)
    np.testing.assert_array_equal(np.asarray(s.materialize(fn(a, axis=axis))), fn(DN, axis=axis))


def test_std_var_ddof_matches_numpy() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", D2)
    assert float(np.asarray(s.materialize(a.std(ddof=1)))) == pytest.approx(np.std(D2, ddof=1))
    assert float(np.asarray(s.materialize(a.var(ddof=1)))) == pytest.approx(np.var(D2, ddof=1))


def test_any_all_on_booleans_match_numpy() -> None:
    s = Session(NumpyBackend())
    b = from_array(s, "b", DB)
    for axis in (None, 0, 1):
        np.testing.assert_array_equal(np.asarray(s.materialize(b.any(axis=axis))), DB.any(axis=axis))
        np.testing.assert_array_equal(np.asarray(s.materialize(b.all(axis=axis))), DB.all(axis=axis))


@pytest.mark.parametrize(
    "fn",
    [np.cumsum, np.cumprod, np.nancumsum, np.nancumprod],
    ids=["cumsum", "cumprod", "nancumsum", "nancumprod"],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_scans_match_numpy(fn: np.ufunc, axis: int | None) -> None:
    s = Session(NumpyBackend())
    data = DN if "nan" in fn.__name__ else D2
    a = from_array(s, "a", data)
    np.testing.assert_array_equal(np.asarray(s.materialize(fn(a, axis=axis))), fn(data, axis=axis))


def test_reduction_forms_track_the_partitioned_axis() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", D2)
    assert a.sum().shape == ()  # global: scalar
    assert a.sum(axis=0).shape == (3,)  # partitioned axis consumed: concrete
    assert a.sum(axis=1).shape == (None,)  # partition-local: axis 0 preserved
    assert a.sum(axis=1, keepdims=True).shape == (None, 1)
    assert a.argmin().dtype == np.dtype(np.intp)
    assert np.cumsum(a).shape == (None,)  # flattening scan: 1-D, unknown length
    assert np.cumsum(a, axis=1).shape == (None, 3)


def test_reductions_on_text_fail_at_record_time() -> None:
    s = Session(NumpyBackend())
    t = from_array(s, "t", np.array(["x", "y"]))
    with pytest.raises(GraphedTypeError):
        t.sum()
    with pytest.raises(GraphedTypeError):
        np.cumsum(t)
