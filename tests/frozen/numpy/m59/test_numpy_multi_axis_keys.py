"""m59 integ-m59-I6 — the numpy backend evaluates the same key kinds with numpy semantics.

`slice`/`int` members inside a tuple key are M13's numpy subscript and stay exactly as they are;
`None` (newaxis) and `Ellipsis` are the kinds m59 adds, and they must evaluate the way numpy does,
stay partition-local, and keep the partitioned axis out of the key.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import graphed.core
from graphed import Array, Session
from graphed.numpy import NumpyBackend, from_array

D2 = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [0.5, 0.25, 0.125], [64.0, 128.0, 256.0]])
D3 = np.arange(24.0).reshape(4, 3, 2)

#: the key kinds m59 adds to the shared surface
ADDED: dict[str, Any] = {
    "[:, :, None]": (slice(None), slice(None), None),
    "[:, None, :]": (slice(None), None, slice(None)),
    "[..., 0]": (Ellipsis, 0),
    "[..., 1:]": (Ellipsis, slice(1, None)),
}

#: what `NumpyArray.__getitem__` accepts today, with the node it records
TODAY: dict[str, tuple[Any, dict[str, str]]] = {
    "[:, 0]": ((slice(None), 0), {"spec": "::,0"}),
    "[:, :2]": ((slice(None), slice(None, 2)), {"spec": "::,:2:"}),
    "[:, ::2]": ((slice(None), slice(None, None, 2)), {"spec": "::,::2"}),
}

REFUSED: dict[str, Any] = {
    "[0, :]": (0, slice(None)),
    "[1:3, 0]": (slice(1, 3), 0),
    "[None, :]": (None, slice(None)),
    "[:, True]": (slice(None), True),
    "[:, 1.5]": (slice(None), 1.5),
    "[..., ..., 0]": (Ellipsis, Ellipsis, 0),
}


def recorded(session: Session, array: Array) -> dict[str, Any]:
    graph = graphed.core.GraphStore.deserialize(session.serialized_ir(array, optimize=False))
    return next(node for node in graph.nodes() if node["id"] == array.node_id)


def loaded() -> tuple[Session, Array, Array]:
    session = Session(NumpyBackend())
    return session, from_array(session, "d2", D2), from_array(session, "d3", D3)


@pytest.mark.parametrize("label", list(ADDED))
def test_newaxis_and_ellipsis_evaluate_as_numpy_does_and_stay_fusible(label: str) -> None:
    key = ADDED[label]
    session, d2, d3 = loaded()
    for deferred, data in ((d2, D2), (d3, D3)):
        out = deferred[key]
        np.testing.assert_array_equal(np.asarray(session.materialize(out)), data[key])
        assert recorded(session, out)["kind"] == "op"  # partition-local, fusible
        assert out.shape == (None, *data[key].shape[1:])


def test_the_tuple_keys_accepted_today_record_the_same_op_params_and_boundary_flag() -> None:
    session, d2, _d3 = loaded()
    for key, params in TODAY.values():
        out = d2[key]
        node = recorded(session, out)
        assert (node["kind"], node["name"], node["params"]) == ("op", "subscript", params)
        np.testing.assert_array_equal(np.asarray(session.materialize(out)), D2[key])


@pytest.mark.parametrize("label", list(REFUSED))
def test_a_tuple_key_that_touches_the_partitioned_axis_or_is_malformed_is_refused(label: str) -> None:
    session, d2, _d3 = loaded()
    before = session.node_count()

    with pytest.raises(TypeError):
        d2[REFUSED[label]]

    assert session.node_count() == before
