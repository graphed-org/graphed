# Test Dispute — `tests/frozen/awkward/m57/test_explain.py::test_explain_reports_the_capstones_fanout_and_independence`

(Shared cause with
`test_explain.py::test_explain_reports_a_weight_registered_before_the_shift_as_reading_moved_objects`
— one fixture, one clause.)

The `hf` assertions PASS. `assert families["pu"].independent_of == {"jes"}` is the dispute.

## The fixture

`_capstone` registers `pu = m57_pu(jets) = 1.0 + 0.25 * gak.num(jets)` on the UNSHIFTED jets and
BEFORE `m57_shifted(ctx, "jes", JES)`, which shifts `Jet` and `MET`. The jet multiplicity no pT shift
moves — but §2.5's diagnostic is per (family, COLLECTION), not per field.

## The clause it contradicts

§2.7 ties independence to §2.5's registry: a weight family flagged there "is neither fanned out nor
independent" (the record's `reads_shifted_by`, the line "reads objects later shifted by"). §2.5's
registry is coarse at the collection level, and that coarseness is pinned by a frozen suite of an
earlier milestone: `tests/frozen/awkward/m49/test_shift_after_weight.py::test_a_weight_registered_before_the_shift_it_reads_is_reported_with_its_collection`
asserts `shift_after_weight == (("btag", "Jet"), ("pu", "MET"))` where `jet_weight` reads
`source.Jet.btag` while the `jes` shift moves only `pt` (`with_field(jets, jets.pt * 1.05, "pt")`).

`m57_pu` reads the `Jet` collection `jes` moves, so the registry flags it exactly as m49 pins
`btag`/`Jet`. Satisfying this assertion needs a FIELD-level §2.5 diagnostic, which m49's frozen suite
forbids.

## The measurement (`scratchpad/p19.py`)

```
weight_first True | §2.5 registry: [('pu', 'Jet')]
   pu: fans_out_over=set() reads_shifted_by={'jes'} independent_of=set() composes_with={'hf'}
   hf: fans_out_over={'jes'} reads_shifted_by=set() independent_of=set() composes_with={'pu'}
weight_first False | §2.5 registry: [('hf', 'Jet'), ('pu', 'Jet')]
   pu: fans_out_over=set() reads_shifted_by={'jes'} independent_of=set() composes_with={'hf'}
   hf: fans_out_over=set() reads_shifted_by={'jes'} independent_of=set() composes_with={'pu'}
```

## Proposed correction

Make the control genuinely independent instead of relying on the diagnostic's blindness: register the
pure weight AFTER the shift, off the nominal collection —
`pu = m57_pu(graphed.nominal(shifted["Jet"]))` registered on the shifted context. Measured
(`scratchpad/p20.py`), every assertion of BOTH tests then passes:

```
weight_first True registry []
   pu: fans=set() reads_shifted_by=set() indep={'jes'} composes={'hf'}
   hf: fans={'jes'} reads_shifted_by=set() indep=set() composes={'pu'}
   'jes' in str(hf): True
weight_first False registry [('hf', 'Jet')]
   pu: fans=set() reads_shifted_by=set() indep={'jes'} composes={'hf'}
   hf: fans=set() reads_shifted_by={'jes'} indep=set() composes={'pu'}
   'jes' in str(hf): True
```

STOPPED on this test.
