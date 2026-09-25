# awkward/m71a — declared External output type, awkward legs (traceability)

Milestone m71 unit A (graphed#57). Authority: the lane plan's `plan-A.md` "Frozen-test contract"
(A-C1, A-C2, A-C3, A-C7, A-C8, the awkward half of A-C5, A-C4's awkward-object legs,
determinism) and the owner's words "all numpy numerical dtypes so float16 should absolutely be
usable".

**New API** (a `TypeError` for the unexpected keyword is the expected failure on a base without it):
`Array.map(..., output_type=)`, `graphed.apply(..., output_type=)`, the backend's optional
`canonical_output_type`, and the `output_type` node-params key.

Run: `python -m pytest tests/frozen/awkward/m71a -q`. Fixture `m71a_awkward_fixtures.py`: a session
whose `AwkwardBackend` carries `BEHAVIOR` (`Photon` → `PhotonArray` with property `pt2` and method
`scaled(k)`, never registered globally) over `events = {run: uint32, x: float32, Jet: {pt: var *
float64}}`. Every declared leg's callable returns a value of the declared type; `same()` compares
values AND types against eager awkward.

| test | contract | pins |
|---|---|---|
| `test_m71a_awkward_declared.py::test_a_declared_bool_mask_indexes_as_a_mask` | A-C1 | `run.map(f, output_type="bool")` → `## * bool`; `~m` bool; `x[m]` records `## * float32` and equals eager |
| `…::test_a_declared_jagged_mask_indexes_as_a_jagged_mask` | A-C1 | `"var * bool"`; `Jet.pt[jm]` records `## * var * float64`, equals eager |
| `…::test_the_declared_type_is_not_a_leaf_cast_of_the_input` | A-C1 | `ak.num(p) > 1` declared `bool` over a jagged input → `## * bool` (a leaf cast gives `var * bool`) |
| `…::test_a_declared_record_exposes_typed_fields` | A-C1 | record → `.eta` is `## * float32` |
| `…::test_a_declared_option_of_records_exposes_optional_fields` | A-C1 | `var * ?{pt, idx}` → `.idx` is `## * var * ?int64` |
| `…::test_a_scalar_input_records_the_declared_scalar` | A-C1 | `gak.sum(x).map(..., "bool")` → `bool` |
| `…::test_every_universe_of_a_varied_apply_records_the_declared_type` | A-C1 | all three `graphed.labels` universes record `## * bool` (a Varied path that drops the keyword records `float32`) |
| `…::test_a_declared_named_record_resolves_its_behavior` | A-C3 | `var * Photon[...]`; `g.pt2`, `g.scaled(3)` record `## * var * float32` and equal eager; projection of `g.pt2` reads exactly `Jet.pt` |
| `test_m71a_awkward_identity.py::test_declared_and_undeclared_are_two_nodes_with_two_forms` | A-C2 | two nodes; undeclared keeps `## * var * float64` and no `output_type` param |
| `…::test_two_declared_types_are_two_nodes` | A-C2 | `var * bool` vs `var * int8` |
| `…::test_spellings_of_one_type_are_one_node_carrying_the_canonical_string` | A-C2 | `"var*bool"`, `"var * bool"`, `Type`, `Form` → one node; params and IR carry `var * bool`, IR never `var*bool` |
| `…::test_each_spelling_family_is_one_node_with_the_canonical_string` | A-C7 | every family in `FAMILIES` (float32, bool, int64, float64, complex128, string, bytes, float16, `3 * float32`, record incl. structured dtype + `ScalarType`, `Photon`, `var * float32`, option-list-record, `ArrayType`/`.content`/`str`) → one node, canonical params + IR + form; string/bytes/float16/regular/record families equal eager |
| `…::test_spelling_families_are_pairwise_distinct_nodes` | A-C7 | families never collapse |
| `…::test_a_python_int_is_int64_on_every_platform` | A-C7 | `int` → the literal `"int64"` |
| `…::test_every_numerical_dtype_is_declarable` | owner (all numerical dtypes, float16) | 14 fixed-width numerical dtypes: name / `np.dtype` / scalar type → one node, params = name, `## * name`, equals eager `values_astype`; pairwise distinct |
| `…::test_declared_recordings_serialize_byte_identically_across_sessions` | Determinism | A-C1/A-C2/A-C3/A-C7 recordings, two fresh sessions, identical bytes |
| `test_m71a_awkward_refusals.py::test_a_type_awkward_cannot_build_is_refused_at_the_call[*]` | A-C5 | `"nope"`, `"var * nope"`, `object`, `"V8"`, `">f4"`, `3`, `"unknown"` → located `GraphedTypeError`, `repr(spec)` in `detail` |
| `…::test_a_scalar_input_refuses_a_record_element` | A-C5 | `"{a: float32}"` over `gak.sum(x)`, located, `repr` in `detail` |
| `…::test_numpy_reads_an_awkward_primitive_as_its_dtype` | A-C4 (awkward objects on numpy) | `NumpyType("float32")` and its `Form` are the node of `"float32"` |
| `…::test_numpy_refuses_an_awkward_list_type_as_it_refuses_the_string` | A-C4 | `ListType(NumpyType("float32"))` refused, located, `detail` equal to that of `"var * float32"` |
| `…::test_a_float16_record_field_records_or_is_refused_by_the_installed_parser` | A-C8 | `[("pt", "f2")]`: records `## * {pt: float16}` and equals eager if awkward parses `var * float16`, else the located A-C5 refusal |
| `…::test_a_jagged_float16_records_or_is_refused_by_the_installed_parser` | A-C8 | `"var * float16"`: same two branches |
