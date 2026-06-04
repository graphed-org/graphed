"""Every node maps to a user-code line, never a graphed* line; the mass node maps to its line."""

from __future__ import annotations

import analyses as analyses_mod
import numpy as np
from analyses import ADL, record
from graphed import Session
from graphed_corpus import make_events

from graphed_awkward import AwkwardBackend, from_awkward


def test_every_node_maps_to_user_code_not_graphed_internals() -> None:
    s, _ = record(ADL["q5"])
    provs = list(s._provenance.values())
    assert provs
    for p in provs:
        # nodes map to the analyses module or the test, never into the graphed* package source
        assert "/src/graphed/" not in p.filename
        assert "/src/graphed_awkward/" not in p.filename
    # at least one node maps to the analyses module (the user's code)
    assert any(p.filename == analyses_mod.__file__ for p in provs)


def test_mass_node_maps_to_its_exact_source_line() -> None:
    import sys

    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=200))
    mu = events.Muon
    pairs = __import__("graphed_awkward").gak.combinations(mu, 2, fields=["a", "b"])
    a, b = pairs.a, pairs.b
    line = sys._getframe().f_lineno + 1
    mass = np.sqrt(2 * a.pt * b.pt * (np.cosh(a.eta - b.eta) - np.cos(a.phi - b.phi)))
    p = s.provenance(mass)
    assert p.filename == __file__
    assert p.lineno == line
    assert "sqrt" in p.source
