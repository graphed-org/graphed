"""Coverage ratchet: ``_helpers.py`` call-template error branches no frozen test ever takes.

Every frozen M27 caller feeds ``parse_call_template`` well-formed args, so the malformed-input
guards (bad JSON, an out-of-range slot, a disallowed group, an unrecognized entry shape) and
``ml_matrix``'s single-slot / rejected-constant branches never run.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphed.preserve import PreserveError
from graphed.preserve.externals._helpers import ml_matrix, parse_call_template


def test_malformed_json_call_template_is_rejected() -> None:
    with pytest.raises(PreserveError, match="not valid JSON"):
        parse_call_template({"args": "{not json"}, 1)


def test_out_of_range_slot_is_rejected() -> None:
    with pytest.raises(PreserveError, match="out of range"):
        parse_call_template({"args": ["$5"]}, n_inputs=2)


def test_group_entry_rejected_when_the_plugin_disallows_groups() -> None:
    # mirrors correctionlib_external's real allow_groups=False call
    with pytest.raises(PreserveError, match="stacked groups"):
        parse_call_template({"args": [["$0", "$1"]]}, n_inputs=2, allow_groups=False)


def test_unrecognized_entry_shape_is_rejected() -> None:
    with pytest.raises(PreserveError, match="not a slot, group, or allowed constant"):
        parse_call_template({"args": [{"unexpected": True}]}, n_inputs=1)


def test_ml_matrix_stacks_a_single_slot_entry() -> None:
    x = np.array([1.0, 2.0, 3.0])
    out = ml_matrix(("slot", 0), [x])
    assert out.shape == (3, 1)
    assert out.dtype == np.float32
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_ml_matrix_rejects_a_constant_entry() -> None:
    # a constant (e.g. correctionlib's systematic-name string) is never a valid model input matrix
    with pytest.raises(PreserveError, match="not valid model inputs"):
        ml_matrix(("const", "up"), [])
