# Required Operations Catalog (M0.5)

Distilled from the A.8 reference corpus. Each entry names an operation/pattern the `graphed`
frontend must support, the corpus analysis it is exercised by, and the **milestone** that must
deliver it. "Extract, do not invent": every row below is exercised by a runnable fixture in
`src/graphed_corpus/` (the catalog and the fixtures are kept in lock-step by
`tests/frozen/m05/test_catalog.py`).

> **Fixture data provenance.** The fixtures run on a deterministic *synthetic* NanoAOD-like dataset
> (`dataset.py`), not real CMS Open Data. This is sufficient for `graphed`'s contract — "same
> answer as plain awkward on the same input" — and keeps the suite network-free. Substituting a
> reduced real AGC/NanoAOD slice (and a CMS-published reference) is a tracked **Phase-2** follow-up.

## A. Array / record operations

| Op | Meaning | Exercised by | Milestone |
|----|---------|--------------|-----------|
| record/field access (`events.Jet.pt`) | columnar attribute access | all | M3 |
| jagged masking (`jets[jets.pt>30]`) | per-element boolean select | adl_q3, adl_q7, ttbar_*, ttgamma_* | M3 |
| `ak.num` | per-event multiplicity | adl_q4, adl_q6, adl_q8, ttbar_*, ttgamma_* | M3 |
| `ak.sum` (axis=1) | per-event reduction (HT) | adl_q7, ttbar_* | M3 |
| `ak.any` / `ak.all` (axis) | per-event boolean reduction | adl_q5, adl_q7, adl_q8 | M3 |
| `ak.combinations` | object pair/triple combinatorics | adl_q5, adl_q6, adl_q8 | M3/M4 |
| `ak.cartesian` (nested) | jet×lepton cross product (isolation) | adl_q7 | M3/M4 |
| `ak.argmin` (keepdims) + gather | "closest-to-mass" selection | adl_q6, adl_q8 | M4 |
| `ak.argsort` / `ak.firsts` | leading-object selection | adl_q8, ttgamma_* | M3 |
| `ak.with_field` / `ak.zip` | build/augment records (flavor, JES) | adl_q8, systematics | M3 |
| `ak.concatenate` (axis=1) | merge collections (leptons) | adl_q7, adl_q8 | M3 |
| `ak.where` / `ak.fill_none` / `ak.drop_none` | option handling | adl_q7, adl_q8, ttgamma_* | M3 |
| elementwise arithmetic + `numpy` ufuncs | kinematics (cos/sin/sqrt/hypot) | adl_q5–q8, systematics | M3 |

## B. Physics / analysis constructs

| Construct | Meaning | Exercised by | Milestone |
|-----------|---------|--------------|-----------|
| invariant mass | 2-/3-body mass from (pt,eta,phi,mass) | adl_q5, adl_q6, adl_q8 | M3 |
| ΔR / Δφ | angular separation, isolation | adl_q7, adl_q8 | M3 |
| transverse mass mT | MET + lepton | adl_q8 | M3 |
| object selection | pt/eta/id cuts → "good" objects | adl_q3, ttbar_*, ttgamma_* | M3 |
| region/category split | 4j1b vs 4j2b; channel selection | ttbar_*, ttgamma_* | M3/M7 |
| 1D histogram fill | count + weighted | all | M3/M7 |

## C. Systematics, corrections, ML

| Pattern | Meaning | Exercised by | Milestone |
|---------|---------|--------------|-----------|
| **weight systematic** | reweight without changing selection (b-tag/photon SF up/down) | ttbar_*_btag_*, ttgamma_pho_* | M7 |
| **kinematic systematic** | JES/JER shift that **re-runs selection** + observables | ttbar_*_jes_*, ttgamma_jes_* | M7 |
| process × variation axis | the AGC histogram layout | ttbar_*, ttgamma_* | M7 |
| correctionlib scale factor | SF from a content-hashed JSON (here: a stand-in fn) | ttbar_* (b-tag), ttgamma_* (photon) | M3 (External node), M9 (payload) |
| ONNX ML inference | model eval as an External node | *(catalogued; fixture is Phase-2 — needs a real/onnx model)* | M7 / M9 |
| CartesianSelection / >64 categories | beyond coffea PackedSelection limit (PocketCoffea) | *(catalogued; Phase-2 executor constraint)* | M7 (Phase 2) |

## Canonical analyses (fixtures with stored references)

- **ADL queries 1–8** — `analyses/adl.py` → `corpus/references/adl_q{1..8}.json`. The graded ladder
  for M3/M4/M5 (column histogram → MET cuts → object selection → combinatorics → 3-lepton mT):
  - `adl_q1` — MET histogram.
  - `adl_q2` — pt of all jets.
  - `adl_q3` — pt of jets with |eta| < 1.0.
  - `adl_q4` — MET for events with ≥2 jets pt>40.
  - `adl_q5` — MET for events with an opposite-sign muon pair, mass in [60,120].
  - `adl_q6` — pt of the trijet system with mass closest to 172.5.
  - `adl_q7` — scalar sum of pt of jets (pt>30) isolated from leptons (ΔR>0.4).
  - `adl_q8` — 3-lepton OSSF transverse mass mT(MET, lead non-pair lepton).
- **AGC ttbar slice** — `analyses/systematics.py::ttbar_region`, regions {4j1b, 4j2b} × variations
  {nominal, jes_up, jes_down, btag_up, btag_down} → `corpus/references/ttbar_*.json`. Drives M7/M9.
- **TTGamma slice** — `analyses/systematics.py::ttgamma_region`, variations {nominal, jes_±, pho_±}
  → `corpus/references/ttgamma_*.json`.

## Phase-2 (catalogued, not built in the MVP fixtures)

- Real AGC/NanoAOD data slice + CMS-published reference (replacing the synthetic dataset).
- ONNX ttbar-reconstruction inference fixture (needs a real model file → External/PayloadDescriptor).
- Systematics-as-a-graph-axis (named axes / template instantiation) — cf. RDataFrame `Vary`.
- CartesianSelection / >64-category selection as a real executor stress fixture.
