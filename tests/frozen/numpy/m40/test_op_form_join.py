"""M40 — ``NumpyBackend.op_form("join")`` record-form inference (§B-7, spec IMPLEMENTATION TARGET 7).

A join's output form is the flat relational record: the union of both sides' fields with the shared
join key kept ONCE. ``how="left"``/``"outer"`` add nullability — and since ``NumpyForm`` has no option
field (forms.py), the validity is carried as a **companion mask column**, i.e. extra fields in the
form. So a nullable join form has strictly MORE fields than the inner form; a how-ignoring impl that
returns the same fields regardless cannot represent the a3 nulls and is caught here.
"""

from __future__ import annotations

import numpy as np

from graphed.numpy import NumpyBackend, NumpyForm

_KEY = "__joinkey__"
_LEFT = NumpyForm(np.dtype(object), kind="record", fields=((_KEY, "<u8"), ("lx", "<i8")))
_RIGHT = NumpyForm(np.dtype(object), kind="record", fields=((_KEY, "<u8"), ("rx", "<f8")))


def _form(be: NumpyBackend, how: str) -> NumpyForm:
    return be.op_form("join", [_LEFT, _RIGHT], {"on": [_KEY], "how": how})


def test_inner_join_form_is_union_of_fields_minus_duplicate_key() -> None:
    be = NumpyBackend()
    form = _form(be, "inner")
    assert form.fields is not None
    names = [f for f, _ in form.fields]
    assert names.count(_KEY) == 1  # the shared key appears ONCE, not twice
    # inner has no nulls ⇒ no validity companions: the field set is exactly the union.
    assert set(names) == {_KEY, "lx", "rx"}


def test_left_and_outer_forms_record_nullability_as_extra_fields() -> None:
    be = NumpyBackend()
    inner = _form(be, "inner")
    left = _form(be, "left")
    outer = _form(be, "outer")
    assert inner.fields is not None and left.fields is not None and outer.fields is not None
    # nullability is carried by companion mask columns ⇒ strictly more fields than the null-free inner.
    assert len(left.fields) > len(inner.fields)
    assert len(outer.fields) > len(inner.fields)
    # the data fields themselves are still all present.
    assert {_KEY, "lx", "rx"} <= {f for f, _ in left.fields}
    assert {_KEY, "lx", "rx"} <= {f for f, _ in outer.fields}
