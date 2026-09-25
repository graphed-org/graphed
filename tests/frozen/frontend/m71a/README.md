# frontend/m71a — declared External output type, numpy legs (traceability)

Milestone m71 unit A (graphed#57). Authority: the lane plan's `plan-A.md` "Frozen-test contract"
(A-C4, the numpy half of A-C5, determinism) and the owner's words "all numpy numerical dtypes so
float16 should absolutely be usable". Only numpy is imported here: this directory is collected by
the numpy-only free-threaded CI job.

**New API** (a `TypeError` for the unexpected keyword is the expected failure on a base without it):
`Array.map(..., output_type=)`, `graphed.apply(..., output_type=)`,
`Session.record_external(..., output_type=)`, and the backend's optional `canonical_output_type`.

Run: `python -m pytest tests/frozen/frontend/m71a -q`. `ListBackend` comes from `frontend/m2`
(`backends.py`, on the pytest `pythonpath`).

| test | contract | pins |
|---|---|---|
| `test_m71a_numpy_declared.py::test_a_float64_declaration_in_every_spelling_is_one_node` | A-C4 | `"f8"`, `"float64"`, `np.float64`, `np.dtype("f8")`, `float` → one node, params `output_type == "float64"`, form `vector[float64]`, value equals eager; `.reduce("sum")` records a scalar |
| `…::test_each_dtype_family_is_one_node_and_families_are_distinct` | A-C4 | bool / int64 / `<U5` / `S5` / `str` (`<U0`) / `object` families: one node each, pairwise distinct, canonical params; `object` is not the undeclared node; undeclared stays `vector[object]` with no `output_type` param |
| `…::test_every_numerical_dtype_is_declarable[*]` | A-C4 + owner (all numerical dtypes, float16) | for each of bool, int8–int64, uint8–uint64, float16/32/64, complex64/128: name / `np.dtype` / scalar type → one node, params = name, form `vector[name]`, value has that dtype and equals eager |
| `…::test_numerical_dtypes_are_pairwise_distinct_nodes` | A-C4 / identity invariant | same name, same input, differing only in declared dtype → never one node |
| `…::test_a_structured_dtype_records_a_record_whose_fields_are_typed` | A-C4 | list and `np.dtype` spellings one node, `record[pt,eta]`, `["eta"]` is `vector[float32]` and equals eager |
| `…::test_a_subarray_dtype_records_a_trailing_shape` | A-C4 | `("f4", (3,))` → `vector[float32, shape=(None, 3)]`, value shape `(2, 3)` |
| `…::test_apply_scalar_2d_and_record_inputs_take_the_declared_element` | A-C4 | `graphed.apply` → `vector[bool]`; scalar input → `scalar[bool]` / `scalar[<U5]`; 2-D input → `vector[bool]`; `from_record` input → `vector[bool]` (not `record[bool]`), values equal eager |
| `…::test_declared_recordings_serialize_byte_identically_across_sessions` | Determinism | two fresh sessions → identical `serialized_ir` bytes containing the declared strings |
| `test_m71a_numpy_refusals.py::test_a_type_numpy_cannot_represent_is_refused_at_the_call[*]` | A-C5 | `"var * float32"`, `"{pt: float32}"`, `"nope"`, `3` → `GraphedTypeError` at the calling line, `repr(spec)` in `detail` |
| `…::test_a_nested_structured_dtype_is_refused` | A-C5 | `"plain dtypes"` in `detail`, located |
| `…::test_a_scalar_input_refuses_fields_and_subarrays[*]` | A-C5 | `"scalar input"` in `detail`, located |
| `…::test_a_backend_without_the_hook_refuses_any_declaration` | A-C5 | `ListBackend` (no `canonical_output_type`) → `"output_type"` in `detail`, located |
| `…::test_output_type_and_an_explicit_form_are_exclusive` | A-C5 | `descriptor=`+`form=`+`output_type=` → `"exclusive"` in `detail`, located |
