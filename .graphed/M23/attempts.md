# M23 attempts — graphed (caller-supplied External descriptors/forms; the graphed-histogram seam)

## Iteration 0 — 2026-06-10 (freeze-M23-0)

- P0.1 (user-confirmed): graphed-histogram records histogram fills as an External FAMILY
  (the M3 correctionlib/ONNX pattern) without teaching backends any histogram content.
- frozen suite tests/frozen/m23 (4 tests); NON-VACUOUS (3/4 fail pre-impl; the no-override
  back-compat pin passes by design).
- record_external gains descriptor=/form= (given together or not at all); when supplied the
  backend is not consulted. Everything downstream unchanged: hash-consed identity, materialize
  with every input, evaluate_ir by content hash.
- gates: 192/192 · coverage >=90 · ruff/mypy/sphinx clean.
