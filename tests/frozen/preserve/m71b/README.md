# preserve/m71b — `record_external(output_type=)` (traceability)

Milestone m71 unit B, the owner's ask (graphed#57): "record_external needs to be extended so it can
specify output type". Authority: the lane plan's `plan-B.md` "Frozen-test contract".

**New API** (a `TypeError` for the unexpected keyword is the expected failure on a base without it):
`graphed.preserve.externals.record_external(..., output_type=)`.

Run: `python -m pytest tests/frozen/preserve/m71b -q` (or the whole preserve subtree). Fixture
`m71b_fixtures.py`: `LUMI` (bool per event over uint32 `run`), `JSEL` (jagged bool over `Jet.pt`),
`PHO` (`var * Photon[pt: float32, eta: float32]` over `Jet.pt`), `MASK` (records every `params` it
is evaluated with in `SEEN`), and a `Photon` behavior known to the backend only.

| test | contract | pins |
|---|---|---|
| `test_m71b_record_external.py::test_a_declared_lumi_mask_is_a_bool_mask` | B-C1 | `## * bool`, `~m` bool, `x[m]` records `## * float32` and equals eager |
| `…::test_a_declared_jagged_selection_indexes_its_jets` | B-C1 | `"var * bool"`; `Jet.pt[jm]` records `## * var * float64`, equals eager |
| `…::test_a_scalar_input_records_the_declared_scalar` | B-C1 | `bool` over `gak.sum(x)` (form only) |
| `…::test_a_declared_named_record_resolves_its_behavior` | B-C1 | `var * Photon[...]`; `g.pt2` records `## * var * float32`, equals eager |
| `…::test_the_declared_type_is_identity_and_the_undeclared_form_is_unchanged` | B-C2 | declared vs undeclared → two nodes; undeclared keeps `## * uint32`, no `output_type` param; `int8` a third node |
| `…::test_spellings_of_one_declared_type_are_one_node` | B-C2 | `"bool"`/`bool`/`np.bool_` one node with params `"bool"`; `"var*bool"`/`"var * bool"`/`ListType` one node; `PHO` string and its `RecordType` object one node |
| `…::test_an_unbuildable_declaration_is_refused_at_the_call[*]` | B-C5 | `"nope"`, `3` → located `GraphedTypeError`, `repr(spec)` in `detail` |
| `…::test_a_numpy_session_still_has_no_payload_descriptor` | B-C5 | numpy session + keyword → `GraphedTypeError` "no payload descriptor" |
| `…::test_declared_recordings_serialize_byte_identically_across_sessions` | Determinism | B-C1/B-C2 recordings, two fresh sessions, identical bytes |
| `test_m71b_bundle.py::test_a_declared_external_reproduces_from_its_bundle` | B-C7 | registered `MASK`; `reproduce(build_bundle(...))` equals the in-process histogram (non-empty); `SEEN`, cleared before `reproduce`, holds only `params["output_type"] == "bool"` from the reproduce run |
| `…::test_the_recorded_evaluator_survives_cloudpickle` | B-C7 | the recorded evaluator round-trips through `cloudpickle` and evaluates equal |
