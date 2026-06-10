# M1 frozen suite — traceability

Maps each M1 Acceptance Contract clause to the tests that cover it.

| Acceptance Contract clause | Tests |
|----------------------------|-------|
| Identical structure → identical NodeId; distinct otherwise | `test_interning.py::test_identical_ops_intern_to_one_node`, `::test_distinct_structure_distinct_ids`, `::test_input_order_matters`, `::test_param_value_and_type_matter` |
| Float edge cases 0.0 / -0.0 / NaN handled deterministically | `test_interning.py::test_float_zero_signed_are_distinct`, `::test_nan_interns_to_itself`, `::test_nan_distinct_from_zero`, `::test_float_and_int_keys_distinct` |
| `node_count` == number of distinct structural keys (property) | `test_property.py::test_node_count_equals_distinct_keys` |
| Concurrent overlapping builds (GIL + 3.14t) → distinct-key count, no panic/race | `test_threadsafe.py::test_concurrent_overlapping_builds_intern_consistently`, `::test_concurrent_distinct_builds_count_exactly` |
| `to_dot()` byte-stable across runs (determinism) | `test_determinism_and_guardrails.py::test_to_dot_is_byte_stable`, `::test_to_dot_is_nonempty_digraph` |
| External nodes intern by full PayloadDescriptor; any field change → distinct id | `test_external.py::*` |
| Guardrail: graph in Rust; `graphed-core` must not import awkward | `test_determinism_and_guardrails.py::test_core_does_not_import_awkward`, `::test_version_returns_string` |
| `reduce(outputs=)` validates node ids; `mark_output` removed (freeze-M22-1) | `test_determinism_and_guardrails.py::test_reduce_outputs_accepts_valid_and_rejects_invalid` |

Rust-side: `cargo test` covers the intern table, total-order float hashing, and a multi-thread
stress test of the locking discipline; a `loom` model of the intern critical section is provided
under `--cfg loom` (see `src/store.rs`).
