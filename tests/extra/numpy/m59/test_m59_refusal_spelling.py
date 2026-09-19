"""m59 extra — the chained spelling a refusal hands back is the key the user wrote.

`a[1:, 0]` is refused because the slice consumes the partitioned axis, and the message offers
`a[<slice>][:, ...]`. That `<slice>` must re-subscript to the SAME node the original slice records:
trimming trailing colons turns `1::` into `1`, an integer index, which is a different op.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphed import Session
from graphed.numpy import NumpyBackend, from_array

D2 = np.arange(12.0).reshape(4, 3)

SLICES = [
    slice(1, 3),
    slice(1, None),
    slice(2, None),
    slice(None, 3),
    slice(None, None, 2),
    slice(1, None, 2),
]


@pytest.mark.parametrize("axis0", SLICES, ids=[str(s) for s in SLICES])
def test_the_offered_spelling_records_the_slice_that_was_refused(axis0: slice) -> None:
    session = Session(NumpyBackend())
    a = from_array(session, "a", D2)

    with pytest.raises(TypeError) as excinfo:
        a[axis0, 0]

    spelling = str(excinfo.value).split("a[", 1)[1].split("][", 1)[0]
    assert eval(f"a[{spelling}]").node_id == a[axis0].node_id
