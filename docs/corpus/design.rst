What the reference suite contains
=================================

When ``graphed`` says a result is reproduced *bit-for-bit*, the question is: against what?
The answer is this suite. It contains no ``graphed`` code at all — its analyses are written
in plain awkward and numpy, so a disagreement is unambiguous: if ``graphed`` and the
plain-awkward version differ on the same input, ``graphed`` is wrong.

What a check looks like
-----------------------

The suite lives under ``tests/_corpus/`` in a source checkout. A check is three steps:
build the deterministic input, run the plain-awkward analysis, compare against the
committed reference — exactly:

.. code-block:: python

    import sys
    from pathlib import Path

    sys.path.insert(0, "tests/_corpus")  # the reference suite ships with the source checkout

    from graphed_corpus import ADL_QUERIES, load_reference, make_events
    from graphed_corpus.histograms import reference_record

    events = make_events()                 # deterministic synthetic events
    h = ADL_QUERIES["q5"](events)          # plain awkward, no graphed imports

    fresh = reference_record(h)
    stored = load_reference(Path("tests/_corpus/references/adl_q5.json"))
    print(fresh == stored)
    print(stored["fingerprint"])

.. code-block:: text

    True
    5c4d46ec76d50fa2

``graphed``'s own tests run the same analyses through record, reduce, and execute, then
assert equality against the same stored counts.

The pieces
----------

**A deterministic synthetic dataset.** ``make_events(n_events=20_000, seed=1234)`` builds a
NanoAOD-shaped jagged record array — electrons, muons, jets with kinematics, charges,
b-tags, MET — from a seeded generator. The same ``(n_events, seed)`` produces the same
arrays on every platform, which is what lets reference outputs be *stored* rather than
regenerated. Synthetic input also keeps the checks network-free and license-free; the
contract being tested is "same answer as plain awkward on the same input", and any
deterministic input exercises that fully. Reading real ROOT files is tested separately —
this suite checks semantics, not I/O.

**The analyses.** Two families, both plain awkward:

* ``analyses/adl.py`` — the eight ADL benchmark queries (``ADL_QUERIES``), a standard
  ladder of HEP query patterns from a bare MET histogram up through combinatorics,
  ΔR cleaning, and a three-lepton transverse mass.
* ``analyses/systematics.py`` — ttbar- and ttgamma-style region selections
  (``ttbar_region``, ``ttgamma_region``) with weight and JES variations sharing most of
  their substructure: the shape that historically made task graphs explode (see
  :doc:`graph_bloat_note`).

**Stored references with fingerprints.** Each analysis's bin counts are committed as JSON
alongside a SHA-256 fingerprint of the record. Changing a reference is a reviewable diff,
never a silent regeneration.

Why exact counts work across platforms
--------------------------------------

Comparing floating-point histograms usually forces tolerances. The suite avoids them with
one safeguard: derived float quantities are rounded to a fixed precision *before* any cut
or fill decision, so a last-ULP difference between platforms cannot flip a value across a
bin edge. References then compare exactly — integer counts for unweighted fills,
fixed-precision values for weighted ones — with no tolerances anywhere.

The operations catalog
----------------------

:doc:`requirements/ops_catalog` lists the awkward operations and analysis patterns the
suite exercises — useful as a quick answer to "does the analysis surface cover the thing
I do?".
