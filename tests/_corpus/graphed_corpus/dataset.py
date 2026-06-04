"""Deterministic synthetic NanoAOD-like event dataset.

The canonical analyses (M0.5) need *runnable* fixtures with stored reference outputs so later
milestones can assert ``graphed`` reproduces plain awkward bit-for-bit. We deliberately use a
seeded synthetic dataset rather than downloading real CMS Open Data: the reproducibility contract
``graphed`` must satisfy is "same answer as plain awkward on the same input", which a deterministic
synthetic input exercises fully, while staying network-free and CI-portable. Replacing this with a
reduced real AGC/NanoAOD slice is a documented follow-up (see docs/requirements/ops_catalog.md).
"""

from __future__ import annotations

import awkward as ak
import numpy as np

# Fields and event shape are fixed so the dataset (and every reference output) is reproducible.
DEFAULT_N_EVENTS = 20_000
DEFAULT_SEED = 1234


def _jagged(rng: np.random.Generator, counts: np.ndarray, **flat_fields: np.ndarray) -> ak.Array:
    """Build a jagged record array from per-event ``counts`` and flat per-object fields."""
    return ak.unflatten(ak.Array(dict(flat_fields)), counts)


def _kin(rng: np.random.Generator, n: int, *, pt_scale: float, eta_max: float) -> dict[str, np.ndarray]:
    return {
        "pt": rng.exponential(pt_scale, n).astype(np.float64) + 5.0,
        "eta": rng.uniform(-eta_max, eta_max, n).astype(np.float64),
        "phi": rng.uniform(-np.pi, np.pi, n).astype(np.float64),
    }


def make_events(n_events: int = DEFAULT_N_EVENTS, seed: int = DEFAULT_SEED) -> ak.Array:
    """Generate a deterministic event record array with Muon/Electron/Jet/Photon/MET collections."""
    rng = np.random.default_rng(seed)

    n_mu = rng.poisson(0.9, n_events)
    mu = _kin(rng, int(n_mu.sum()), pt_scale=20.0, eta_max=2.4)
    mu["charge"] = (rng.integers(0, 2, mu["pt"].size) * 2 - 1).astype(np.int64)
    mu["mass"] = np.full(mu["pt"].size, 0.105_658, dtype=np.float64)

    n_el = rng.poisson(0.7, n_events)
    el = _kin(rng, int(n_el.sum()), pt_scale=20.0, eta_max=2.5)
    el["charge"] = (rng.integers(0, 2, el["pt"].size) * 2 - 1).astype(np.int64)
    el["mass"] = np.full(el["pt"].size, 0.000_511, dtype=np.float64)

    n_jet = rng.poisson(3.5, n_events)
    jet = _kin(rng, int(n_jet.sum()), pt_scale=30.0, eta_max=4.5)
    jet["mass"] = (rng.exponential(8.0, jet["pt"].size) + 1.0).astype(np.float64)
    jet["btag"] = rng.uniform(0.0, 1.0, jet["pt"].size).astype(np.float64)

    n_pho = rng.poisson(0.5, n_events)
    pho = _kin(rng, int(n_pho.sum()), pt_scale=25.0, eta_max=2.5)

    met_pt = rng.exponential(25.0, n_events).astype(np.float64)
    met_phi = rng.uniform(-np.pi, np.pi, n_events).astype(np.float64)

    return ak.Array(
        {
            "Muon": _jagged(rng, n_mu, **mu),
            "Electron": _jagged(rng, n_el, **el),
            "Jet": _jagged(rng, n_jet, **jet),
            "Photon": _jagged(rng, n_pho, **pho),
            "MET": ak.zip({"pt": met_pt, "phi": met_phi}),
        }
    )
