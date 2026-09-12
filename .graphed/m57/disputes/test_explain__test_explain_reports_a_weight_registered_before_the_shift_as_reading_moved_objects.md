# Test Dispute — `tests/frozen/awkward/m57/test_explain.py::test_explain_reports_a_weight_registered_before_the_shift_as_reading_moved_objects`

Its first two assertions PASS — `hf`, registered before the shift, is reported as reading moved
objects and its line names `jes`. The third,
`assert families["pu"].independent_of == {"jes"}  # the live control: this one IS independent`, is the
same dispute as
[`test_explain_reports_the_capstones_fanout_and_independence`](test_explain__test_explain_reports_the_capstones_fanout_and_independence.md):
one `_capstone` fixture, one clause.

§2.7 ties independence to §2.5's registry, which is per (family, COLLECTION). `m57_pu` reads the
`Jet` collection `m57_shifted` moves, so the registry flags `("pu", "Jet")` in this program too
(measured), and a flagged family is neither fanned out nor independent. The collection-level
coarseness is pinned by
`tests/frozen/awkward/m49/test_shift_after_weight.py::test_a_weight_registered_before_the_shift_it_reads_is_reported_with_its_collection`
(`jet_weight` reads `Jet.btag`; the shift moves only `pt`; it is still reported).

Measurement and the verified correction (register the pure weight after the shift, off the nominal
collection — both tests then pass in full) are in the companion file.

STOPPED on this test.
