"""The recorded graph is semantically correct: materialize matches plain awkward (plan M3).

(Bit-for-bit reproduction through the real executor is M7; here we validate the recorded graph by
the reference node-by-node evaluator.)
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from analyses import ADL
from graphed_corpus.analyses.adl import _pair_mass


def _flat(x: object) -> np.ndarray:
    return ak.to_numpy(ak.flatten(ak.Array(x), axis=None))


def _reference(name: str, ev: ak.Array) -> np.ndarray:
    if name == "q1":
        return ak.to_numpy(ev.MET.pt)
    if name == "q2":
        return _flat(ev.Jet.pt)
    if name == "q4":
        njet = ak.num(ev.Jet[ev.Jet.pt > 40], axis=1)
        return ak.to_numpy(ev.MET.pt[njet >= 2])
    if name == "q5":
        pr = ak.combinations(ev.Muon, 2, fields=["a", "b"])
        m = _pair_mass(pr.a, pr.b)
        keep = ak.any((m > 60) & (m < 120) & (pr.a.charge != pr.b.charge), axis=1)
        return ak.to_numpy(ev.MET.pt[keep])
    raise KeyError(name)


@pytest.mark.parametrize("name", ["q1", "q2", "q4", "q5"])
def test_materialize_matches_plain_awkward(name: str, shared_events: ak.Array) -> None:
    from graphed import Session

    from graphed_awkward import AwkwardBackend, from_awkward

    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", shared_events)
    got = _flat(s.materialize(ADL[name](events)))
    expected = _reference(name, shared_events)
    assert got.shape == expected.shape
    assert np.allclose(np.sort(got), np.sort(expected))
