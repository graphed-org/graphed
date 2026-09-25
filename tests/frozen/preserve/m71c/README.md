# preserve/m71c — built-in plugin and template output types (traceability)

Milestone m71 unit C, the class cut: the float64-valued built-in plugins and the gak correction
template record float64 instead of their first input's type, and every External route refuses a
foreign input at the calling line. Authority: the lane plan's `plan-C.md` "Frozen-test contract"
(C-C3, C-C5, C-C8) and the owner's words "all numpy numerical dtypes so float16 should absolutely be
usable".

**New API** (expected base failures: `TypeError` for the unknown `output_dtype=` field or the
`output_type=` keyword, `AttributeError` for a missing `.output_dtype`, or the old `## * float32`
form): `ExternalPlugin.output_dtype`, `record_external(..., output_type=)`.

Run: `python -m pytest tests/frozen/preserve/m71c -q` (or the whole preserve subtree). Fixture
`m71c_fixtures.py`: byte-literal `CSET` (correctionlib) and `MODEL` (ONNX); `plugin(kind, dtype,
**fields)` builds a custom plugin whose value is its first input cast to `dtype` — only inside test
bodies, since `output_dtype=` is new.

| test | contract | pins |
|---|---|---|
| `test_m71c_plugin_defaults.py::test_the_float64_builtins_declare_float64_and_the_histogram_plugin_nothing` | C-C3 | seven built-ins `.output_dtype == "float64"`, `HISTOGRAM_PLUGIN.output_dtype is None` |
| `…::test_a_correctionlib_external_records_float64_over_a_float32_input` | C-C3 | `## * float64` (value `4 * float64`); scalar input → `float64` (form only) |
| `…::test_the_correction_template_records_float64` | C-C3 | `gak.apply_correction(..., args=)` → `## * float64`; scalar input → `float64` |
| `…::test_an_explicit_output_type_wins_over_the_plugin_default` | C-C3 | `output_type="float32"` → `## * float32`, a different node |
| `…::test_every_float64_spelling_of_a_plugin_default_records_float64[*]` | C-C3 | `float`, `np.float64`, `"f8"` → `## * float64`, equals eager |
| `…::test_a_numpy_session_still_has_no_payload_descriptor` | C-C3 control | passes on the base |
| `…::test_an_unknown_plugin_default_is_refused_at_the_call` | C-C5 | `output_dtype="nope"` → located, `"nope"` in `detail` |
| `…::test_a_non_leaf_plugin_default_is_refused_at_the_call` | C-C5 | `output_dtype="var * float32"` → located, `"output_dtype"` in `detail` |
| `…::test_a_float16_plugin_default_records_float16` | C-C5 + owner | `np.float16` → `## * float16`, equals eager |
| `…::test_every_numerical_plugin_default_records_its_dtype[*]` | owner (all numerical dtypes) | each of the 14 fixed-width numerical dtypes → `## * name`, equals eager |
| `test_m71c_foreign_inputs.py::test_a_plugin_external_refuses_a_foreign_input` | C-C8 | located `"different Session"`; passes on the base |
| `…::test_the_correction_template_refuses_a_foreign_input` | C-C8 | located (plain `TypeError` on the base) |
| `…::test_the_onnx_template_refuses_a_foreign_input` | C-C8 | located (plain `TypeError` on the base) |
