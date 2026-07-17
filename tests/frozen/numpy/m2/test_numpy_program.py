"""A numpy-backend program (map/filter/arith/reduce) builds a small correct graph (M2)."""

from __future__ import annotations

import numpy as np

from graphed import Session
from graphed.numpy import NumpyBackend, NumpyForm, from_array


def test_twenty_line_program_builds_correct_graph_and_evaluates() -> None:
    s = Session(NumpyBackend())
    pt = from_array(s, "pt", np.array([10.0, 40.0, 25.0, 5.0]))
    eta = from_array(s, "eta", np.array([0.1, -2.0, 1.5, 0.3]))
    weight = from_array(s, "weight", np.array([1.0, 1.0, 1.0, 1.0]))
    mask = from_array(s, "mask", np.array([False, True, True, False]))

    scaled = pt * weight  # arith
    shifted = scaled + eta  # arith
    selected = shifted.filter(mask)  # filter (boolean mask)
    total = selected.reduce("sum")  # reduction over the numeric selection
    doubled = selected.map(lambda a: np.asarray(a) * 2, name="double")  # opaque map -> External

    # structure: pt, eta, weight, mask, scaled, shifted, selected, total, doubled = 9 nodes
    assert s.node_count() == 9
    assert isinstance(s.form(total), NumpyForm)

    # scaled=[10,40,25,5]; shifted=[10.1,38,26.5,5.3]; selected(mask)=[38,26.5]
    assert np.isclose(float(s.materialize(total)), 64.5)
    assert np.allclose(s.materialize(doubled), [76.0, 53.0])


def test_form_inference_promotes_dtypes() -> None:
    s = Session(NumpyBackend())
    i = from_array(s, "i", np.array([1, 2, 3]))
    f = from_array(s, "f", np.array([1.5, 2.5, 3.5]))
    assert s.form(i).describe().startswith("vector[int")
    promoted = i + f
    assert "float" in s.form(promoted).describe()


def test_repeated_subexpression_interns() -> None:
    s = Session(NumpyBackend())
    a = from_array(s, "a", np.array([1.0, 2.0]))
    b = from_array(s, "b", np.array([3.0, 4.0]))
    first = a + b
    n = s.node_count()
    second = a + b
    assert second.node_id == first.node_id
    assert s.node_count() == n


def test_structure_independent_of_data_backend_via_dot() -> None:
    # building the same program twice yields byte-identical recorded structure
    def build() -> str:
        s = Session(NumpyBackend())
        a = from_array(s, "a", np.array([1.0, 2.0, 3.0]))
        b = from_array(s, "b", np.array([4.0, 5.0, 6.0]))
        (a + b).reduce("sum")
        return s.to_dot()

    assert build() == build()
