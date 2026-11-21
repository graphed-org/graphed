# Test Dispute — `tests/frozen/preserve/m71b/test_m71b_record_external.py::test_the_declared_type_is_identity_and_the_undeclared_form_is_unchanged`

Status: CLOSED — owner rulings via team-lead: #253 "option 2 with 1 as fallback" and, for the seven issues, "refreezes are authorized"; refrozen in the #253 PR as below, tag freeze-m71-fixup4.

## The pin

An undeclared `record_external(s, LUMI, b"g", [ev.run])` records `## * uint32`, the first input's form, though `LUMI` returns a bool mask.

## The clause it contradicts

graphed#253 option 2: a plugin External with no stated type records the type of one length-zero evaluation where the plugin survives one, checked like a claim, and warns at record time otherwise. `LUMI` survives one and gives `## * bool`.

## Correction

The test is renamed `…_undeclared_form_is_inferred` and asserts `describe(s, undeclared) == "## * bool"`; the README row says the undeclared node records the type its length-zero call gives. The node-identity assertions are unchanged.
