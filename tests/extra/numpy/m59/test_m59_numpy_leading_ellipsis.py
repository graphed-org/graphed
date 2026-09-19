"""m59 extra — the numpy backend states the leading-`...` rule instead of tripping over it.

Its metas are zero-length, so a key that lands on the partitioned axis used to surface as an
`IndexError` from indexing an empty axis — an accident that says nothing about the key, and one a
non-empty meta would not raise at all. The rule is the same typing rule the awkward backend
applies, measured here against `ndim`.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphed import Session
from graphed.errors import GraphedTypeError
from graphed.numpy import NumpyBackend, from_array

D1 = np.arange(4.0)
D2 = np.arange(12.0).reshape(4, 3)


@pytest.mark.parametrize(
    ("data", "key"),
    [(D1, (Ellipsis, 0)), (D1, (Ellipsis, slice(1, None))), (D2, (Ellipsis, 0, 0))],
    ids=["d1[..., 0]", "d1[..., 1:]", "d2[..., 0, 0]"],
)
def test_a_leading_ellipsis_that_absorbs_nothing_is_ill_typed(
    data: np.ndarray, key: tuple[object, ...]
) -> None:
    session = Session(NumpyBackend())
    a = from_array(session, "a", data)
    before = session.node_count()

    with pytest.raises(GraphedTypeError, match=r"leading \.\.\. must absorb the partitioned axis"):
        a[key]

    assert session.node_count() == before


def test_a_leading_ellipsis_with_an_axis_to_absorb_still_evaluates() -> None:
    session = Session(NumpyBackend())
    a = from_array(session, "a", D2)

    for key in ((Ellipsis, 0), (Ellipsis, slice(1, None)), (Ellipsis, None, 0)):
        np.testing.assert_array_equal(np.asarray(session.materialize(a[key])), D2[key])
