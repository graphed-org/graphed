How graphed-corpus works
========================

``graphed-corpus`` is the ecosystem's *requirements made executable*. It contains no framework
code at all — deliberately. What it holds is the ground truth every other package is measured
against: the catalog of operations a real HEP analysis needs, canonical analyses written in
**plain awkward** (the semantics oracle), a deterministic synthetic dataset to run them on,
and stored reference histograms with content fingerprints so "reproduces bit-for-bit" is a
checkable claim rather than a slogan.

When a frozen test elsewhere says "graphed must equal awkward here", *here* is this package.

.. contents::
   :local:
   :depth: 2


The operations catalog
----------------------

``docs/requirements/ops_catalog.md`` is the M0.5 deliverable: the enumerated operations
(selection, combinatorics, kinematics, reductions, corrections, inference, histogramming...)
that the canonical analyses exercise, derived from real analysis patterns (ADL benchmarks, AGC
ttbar, a ttgamma-style selection). Frontends and backends grow against this list — a
capability is "done" when the corpus analyses that need it pass against their references.

The dataset: synthetic on purpose
---------------------------------

``make_events(n_events=20_000, seed=1234)`` builds a NanoAOD-shaped jagged record array —
electrons/muons/jets with kinematics, charges, b-tags, MET — from a seeded generator. The
choice of *synthetic* over downloaded open data is an argument, not a shortcut: the contract
graphed must satisfy is "**same answer as plain awkward on the same input**", and a
deterministic synthetic input exercises that fully while staying network-free, license-free,
and CI-portable. (Real-data fidelity is tested elsewhere — the reader-integration and
benchmark forks run against actual ROOT files; the corpus is the *semantics* oracle, not the
*I/O* oracle.)

Determinism is total: fixed field list, fixed shapes, seeded draws. The same
``(n_events, seed)`` produces the same arrays on every platform — which is what lets reference
outputs be *stored* rather than regenerated.

The canonical analyses
----------------------

Two families, all written in plain awkward + numpy (no graphed imports — the oracle must not
depend on the thing it judges):

* ``analyses/adl.py`` — the eight ADL benchmark queries (``ADL_QUERIES``): MET, jet kinematics
  with cuts, dimuon mass windows, trijet combinatorics, ΔR cleaning, SFOS+MT — the standard
  functionality ladder.
* ``analyses/systematics.py`` — the shapes that stress *graph structure* rather than
  operations: ttbar/ttgamma-style regions (``ttbar_region``, ``ttgamma_region``,
  ``TTBAR_FIXTURES``...) with many shared-substructure variations, the pattern that historically
  produced graph bloat and O(N²) optimization (see ``docs/graph_bloat_note.md`` for the
  measured motivation).

Reference histograms and fingerprints
-------------------------------------

``hist1d`` fills integer-count histograms with one cross-platform safeguard: derived float
quantities are rounded to a fixed precision (``stable``) *before* cut and fill decisions, so a
last-ULP difference between platforms cannot flip a value across a bin edge. References then
compare as **exact integer counts** — no tolerances anywhere.

``write_reference`` / ``load_reference`` store and retrieve each analysis's bin counts
together with a ``fingerprint`` (a SHA-256 over the canonical reference record). A later
milestone — or a rerun on different hardware — asserts equality against the stored counts and
the fingerprint. The references are committed; changing one is a reviewable diff, never a
silent regeneration.

How the rest of the ecosystem uses this
---------------------------------------

The pattern, wherever it appears (the M3 backend suites, the M7 executor end-to-end runs, the
M9 preservation round trips), is always the same three steps::

    events = make_events()                       # the deterministic input
    expected = ADL_QUERIES["q5"](events)         # the plain-awkward oracle
    # ... run the SAME query through graphed (record, reduce, execute) ...
    assert counts_from_graphed == counts_from(expected)   # exact, integer, no tolerance

Because the oracle is plain awkward, a disagreement is unambiguous: graphed is wrong (or
awkward changed underneath both — which version pins surface immediately).


Phase 2 (deliberately not built)
--------------------------------

* **A reduced real-data slice** (AGC/NanoAOD) alongside the synthetic dataset — documented as
  the follow-up in the ops catalog; the synthetic oracle stays regardless.
* **Catalog growth** tracking new ecosystem capabilities (the histogramming and preservation
  ops entered the ladder through the benchmark work; the catalog document trails and should
  absorb them).
* **Weighted/systematic reference families** beyond the current fixtures.

See :doc:`improvements` for the live tracked list.
