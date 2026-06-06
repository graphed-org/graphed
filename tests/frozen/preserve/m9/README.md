# M9 frozen suite — graphed-preserve (analysis preservation bundle)

The fixture (`agc.py`) is a reduced **AGC ttbar slice** recorded through graphed/graphed-awkward with
a **real correctionlib JSON** (weight SF + systematic) and a **real ONNX model** (per-event
inference), exercising JES (kinematic) + correctionlib (weight) systematics — the analysis the plan
chose for M9 precisely because it carries an ONNX model, correctionlib SFs, and systematics.

| Test (file) | Plan M9 clause it pins |
|---|---|
| `test_reproduce.py::test_reproduce_matches_build_bit_for_bit` | "reproduces its histograms bit-for-bit" (from references alone) |
| `…::test_bundle_carries_real_hep_standard_payloads` | reuse HEP standards (correctionlib + ONNX), content-addressed; no opaque nodes |
| `…::test_bundle_is_a_self_contained_directory` | every referenced blob lives in the bundle's own store |
| `…::test_systematic_variations_change_the_result` | the bundle contains real systematics (JES + correctionlib up/down) |
| `test_fingerprint.py::test_identical_inputs_give_an_identical_fingerprint` | "changing nothing reproduces an identical hash" (no over-sensitivity: no timestamps/abs-paths) |
| `…::test_changing_the_correctionlib_json/_onnx_weights/_dataset/_config_..._changes_the_fingerprint` | "changing any auxiliary input changes the top-level content hash" |
| `…::test_a_payload_change_also_changes_the_canonical_ir` | the External descriptor's content hash is in the IR (cache-poisoning defence) |
| `test_no_originals.py::test_reproduce_with_no_original_code_or_inputs` | "bit-for-bit on machine B with NO access to original code/env/inputs" (scrubbed subprocess; payloads deleted) |
| `test_inspect.py::test_inspect_renders_ir_provenance_and_payload_inventory` | `inspect` renders IR + M6 provenance + payload inventory |
| `…::test_inspect_neither_executes_nor_resolves_data` | `inspect` works without executing (data/payloads deleted; reproduce then fails) |
| `…::test_inspect_is_faithful_to_the_reproduced_graph` | `inspect` is faithful to what `reproduce` runs (same IR nodes) |
| `…::test_opaque_cloudpickled_node_is_flagged_as_a_risk` | every `opaque=True` node flagged as a preservation risk |
| `test_missing_payload.py::*` | a missing payload (correction / dataset / IR) fails with "unresolved … <hash>", never silently |
| `test_build_errors.py::*` | build/evaluate fail honestly (missing payload bytes; payload-hash mismatch; unregistered kind) |
| `test_plugins.py::test_builtin_plugins_have_valid_hashes` | the onnx + correctionlib plugins pass cross-process determinism + non-vacuity validation |
| `…::test_correctionlib_hashes_contents_not_formatting`, `…::test_onnx_hashes_weights` | content hash = correctionlib *contents* / ONNX *weights*, not raw file bytes |
| `…::test_vacuous_hash_is_rejected`, `…::test_nondeterministic_hash_is_rejected`, `…::test_time_based_hash_is_rejected` | `register_plugin` rejects a vacuous (constant) or non-deterministic (hash()/time) hash |
| `…::test_user_plugin_registers_and_reproduces_bit_for_bit`, `…::test_user_plugin_fingerprint_tracks_its_payload`, `…::test_reopened_user_bundle_reproduces` | a user-defined External plugin (the template) records, preserves, and reproduces bit-for-bit |

Frozen = read-only after the freeze tag (see `.graphed/M9/`).
