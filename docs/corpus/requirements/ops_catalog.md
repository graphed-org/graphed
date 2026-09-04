# Operations catalog

Each entry names an operation/pattern the `graphed` frontend supports and the corpus analysis it
is exercised by. Every row whose last column names an analysis is exercised by a runnable fixture
in the reference suite (`tests/_corpus/` in a source checkout); two rows in section C are
catalogued without a fixture and say so in that column.

> **Fixture data provenance.** The fixtures run on a deterministic *synthetic* NanoAOD-like dataset
> (`dataset.py`), not real CMS Open Data. This is sufficient for `graphed`'s contract — "same
> answer as plain awkward on the same input" — and keeps the suite network-free.

## A. Array / record operations

| Op | Meaning | Exercised by |
|----|---------|--------------|
| record/field access (`events.Jet.pt`) | columnar attribute access | all |
| jagged masking (`jets[jets.pt>30]`) | per-element boolean select | adl_q3, adl_q7, ttbar_*, ttgamma_* |
| `ak.num` | per-event multiplicity | adl_q4, adl_q6, adl_q8, ttbar_*, ttgamma_* |
| `ak.sum` (axis=1) | per-event reduction (HT) | adl_q7, ttbar_* |
| `ak.any` / `ak.all` (axis) | per-event boolean reduction | adl_q5, adl_q7, adl_q8 |
| `ak.combinations` | object pair/triple combinatorics | adl_q5, adl_q6, adl_q8 |
| `ak.cartesian` (nested) | jet×lepton cross product (isolation) | adl_q7 |
| `ak.argmin` (keepdims) + gather | "closest-to-mass" selection | adl_q6, adl_q8 |
| `ak.argsort` / `ak.firsts` | leading-object selection | adl_q8, ttgamma_* |
| `ak.with_field` / `ak.zip` | build/augment records (flavor, JES) | adl_q8, systematics |
| `ak.concatenate` (axis=1) | merge collections (leptons) | adl_q7, adl_q8 |
| `ak.where` / `ak.fill_none` / `ak.drop_none` | option handling | adl_q7, adl_q8, ttgamma_* |
| elementwise arithmetic + `numpy` ufuncs | kinematics (cos/sin/sqrt/hypot) | adl_q5–q8, systematics |

## B. Physics / analysis constructs

| Construct | Meaning | Exercised by |
|-----------|---------|--------------|
| invariant mass | 2-/3-body mass from (pt,eta,phi,mass) | adl_q5, adl_q6, adl_q8 |
| ΔR / Δφ | angular separation, isolation | adl_q7, adl_q8 |
| transverse mass mT | MET + lepton | adl_q8 |
| object selection | pt/eta/id cuts → "good" objects | adl_q3, ttbar_*, ttgamma_* |
| region/category split | 4j1b vs 4j2b; channel selection | ttbar_*, ttgamma_* |
| 1D histogram fill | count + weighted | all |

## C. Systematics, corrections, ML

| Pattern | Meaning | Exercised by |
|---------|---------|--------------|
| **weight systematic** | reweight without changing selection (b-tag/photon SF up/down) | ttbar_*_btag_*, ttgamma_pho_* |
| **kinematic systematic** | JES/JER shift that **re-runs selection** + observables | ttbar_*_jes_*, ttgamma_jes_* |
| process × variation axis | the AGC histogram layout | ttbar_*, ttgamma_* |
| correctionlib scale factor | SF from a content-hashed JSON (here: a stand-in fn) | ttbar_* (b-tag), ttgamma_* (photon) |
| ONNX ML inference | model eval as an External node | *(catalogued; no fixture yet — needs a real ONNX model)* |
| CartesianSelection / >64 categories | beyond coffea PackedSelection limit (PocketCoffea) | *(catalogued; no fixture yet)* |

## Canonical analyses (fixtures with stored references)

- **ADL queries 1–8** — `analyses/adl.py` → `references/adl_q{1..8}.json`. The graded ladder
  (column histogram → MET cuts → object selection → combinatorics → 3-lepton mT):
  - `adl_q1` — MET histogram.
  - `adl_q2` — pt of all jets.
  - `adl_q3` — pt of jets with |eta| < 1.0.
  - `adl_q4` — MET for events with ≥2 jets pt>40.
  - `adl_q5` — MET for events with an opposite-sign muon pair, mass in [60,120].
  - `adl_q6` — pt of the trijet system with mass closest to 172.5.
  - `adl_q7` — scalar sum of pt of jets (pt>30) isolated from leptons (ΔR>0.4).
  - `adl_q8` — 3-lepton OSSF transverse mass mT(MET, lead non-pair lepton).
- **AGC ttbar slice** — `analyses/systematics.py::ttbar_region`, regions {4j1b, 4j2b} × variations
  {nominal, jes_up, jes_down, btag_up, btag_down} → `references/ttbar_*.json`.
- **TTGamma slice** — `analyses/systematics.py::ttgamma_region`, variations {nominal, jes_±, pho_±}
  → `references/ttgamma_*.json`.

## Not covered by the fixtures yet

- A real AGC/NanoAOD data slice + CMS-published reference (the synthetic dataset stands in).
- An ONNX ttbar-reconstruction inference fixture (needs a real model file).
- A CartesianSelection / >64-category selection stress fixture.
