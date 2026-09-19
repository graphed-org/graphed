"""m59 extra — a wide integer operand stays exact through the DURABLE path.

`session.form`/`session.materialize` read the params still held in this process, so they answer
correctly even when the store has silently widened the value to f64. Only a compile + pickle +
evaluate round trip reads the value back out of the serialized IR, which is where recording a wide
int as text earns its keep: 2**63 + 1 comes back as 9223372036854775808 without it, and
2**64 - 1 does not come back at all.
"""

from __future__ import annotations

import pickle

import awkward as ak
import numpy as np
import pytest

from graphed import Session, compile_ir, evaluate_ir
from graphed.awkward import AwkwardBackend, from_awkward

#: ones, so the product is the scalar itself and the comparison is about the operand alone
ONES = ak.values_astype(ak.Array([[1, 1], [1]]), np.uint64)

WIDE = {"2**63 + 1": np.uint64(2**63 + 1), "2**64 - 1": np.uint64(2**64 - 1)}


@pytest.mark.parametrize("label", list(WIDE))
def test_a_wide_scalar_operand_survives_serialization_exactly(label: str) -> None:
    value = WIDE[label]
    session = Session(AwkwardBackend())
    events = from_awkward(session, "events", ONES)

    compiled = pickle.loads(pickle.dumps(compile_ir(session, events * value)))
    got = evaluate_ir(compiled, AwkwardBackend(), {"events": ONES})

    assert ak.to_list(got[0]) == ak.to_list(ONES * value)
    assert np.asarray(ak.flatten(got[0])).dtype == np.dtype("uint64")
