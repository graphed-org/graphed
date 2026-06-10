# Frozen acceptance suite — M23 (graphed): caller-supplied External descriptors and forms

The seam graphed-histogram records through (P0.1, user-confirmed plan 2026-06-10). Traceability:

| Test file | Verifies | Item |
|---|---|---|
| `test_external_overrides.py` | `record_external(descriptor=, form=)` skips backend consultation (backends stay domain-free, §A.4); without overrides unknown External ops still fail (back-compat pin); the caller's form is returned verbatim; materialize calls the evaluator with EVERY input; `evaluate_ir` resolves by the caller's content hash and fails loudly when unresolved; descriptor+params identity is hash-consed | P0.1 |
