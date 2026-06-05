"""Shared analyses for the M6 suite, including a module-level worker so the process-boundary test can
build + run the failing analysis inside a spawned child process."""

from __future__ import annotations

import graphed_numpy as gn
import numpy as np
from graphed import Session

import graphed_debug as gd


def numpy_oob(n: int = 3):
    """A numpy analysis with an out-of-range index inside an opaque op (stands in for a mass calc)."""
    s = Session(gn.NumpyBackend())
    events = gn.from_record(s, "events", pt=np.arange(1.0, n + 1), eta=np.linspace(0, 1, n))
    scaled = events["pt"] * 2.0
    bad = scaled.map(lambda a: a[100], name="oob_index")  # <-- the failing analysis line
    return s, bad


def numpy_mismatch_in_fused_stage():
    """A shape mismatch buried INSIDE a fused arithmetic stage (a non-boundary op), to prove the
    error maps to the right member deep in a fused kernel — not just the first or boundary op."""
    s = Session(gn.NumpyBackend())
    a = gn.from_record(s, "a", x=np.arange(3.0))["x"]
    b = gn.from_record(s, "b", y=np.arange(5.0))["y"]
    stepped = a * 2.0 + 1.0
    bad = stepped + b  # <-- len-3 + len-5: a non-boundary op deep in the fused stage raises
    return s, bad


def awkward_mass_oob(n_events: int = 200):
    """A dimuon mass calc that indexes the 2nd muon — out of range for the (many) events with fewer
    than two muons (the canonical 'out-of-range index into the mass calc' of the M6 contract). The
    positional index is awkward-specific, so the calc lives in a map; the error surfaces at eval."""
    import numpy as anp  # noqa: PLC0415 - awkward-only helpers, kept off the numpy-backend import
    from graphed_awkward import AwkwardBackend, from_awkward  # noqa: PLC0415
    from graphed_corpus import make_events  # noqa: PLC0415

    def dimuon_mass(mu: object) -> object:
        a, b = mu[:, 0], mu[:, 1]  # <-- out-of-range 2nd-muon index for single-muon events
        return anp.sqrt(2 * a.pt * b.pt * (anp.cosh(a.eta - b.eta) - anp.cos(a.phi - b.phi)))

    s = Session(AwkwardBackend())
    events = from_awkward(s, "events", make_events(n_events=n_events))
    mass = events.Muon.map(dimuon_mass, name="dimuon_mass")  # <-- the mass-calc analysis line
    return s, mass


def run_numpy_oob_and_raise() -> None:
    """Module-level worker for multiprocessing: build + run in THIS process, letting the StageError
    propagate (so the parent receives it pickled across the process boundary)."""
    s, bad = numpy_oob()
    gd.run(s, bad, opt_level=1)
