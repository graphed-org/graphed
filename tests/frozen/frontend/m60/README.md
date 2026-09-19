# frontend/m60 — library-integration seams (traceability)

Authority: `graphed-workdir/m60-decomposition.md` (contract lines X1–X3, O1–O4, V1–V3, E1, one
frozen property each) under root-prompt rule **R25.3**. Four decisions move, all backend-neutral:

* **integ-m60-X** — a `Session` refuses an `Array` it did not record at every method that takes
  one, through ONE guard: the `record_*` family with `GraphedTypeError` carrying `record_op`'s
  message, the reading entries and the module verbs that take a session beside arrays with
  `TypeError`. A node id only means something in its own store.
* **integ-m60-O** — two DISTINCT callable objects recorded without `name=` never share a node: the
  `fn` param carries a per-Session first-seen ordinal on the derived name (`q`, then `q#1`).
  `name=` stays the caller's identity declaration.
* **integ-m60-V** — `graphed.provenance.register_internal(prefix)` makes `capture()` skip a
  wrapping library's frames, matching whole dotted components, so the library's ops — and the
  `StageError` they raise at run time — point at the user's line.
* **integ-m60-E** — `graphed.expand` is public.

Run: `python -m pytest tests/frozen/frontend/m60 -q` (its own process, per the frontend
per-milestone split). The `m60_` prefix is load-bearing under prepend import mode.

The awkward-backed half of the milestone (integ-m60-P) is `tests/frozen/awkward/m60`. Nothing here
imports `awkward`, `pyarrow`, `pandas` or `hist` at module or body level — the free-threaded
frontend job installs only `pytest hypothesis numpy` — and `graphed.debug`, which V3 needs, pulls
none of them either (every optional dependency of its dashboard is imported lazily).

## Fixtures

* `m60_seams.py` — `ToyBackend` types every op and carries data through, so a node's IDENTITY and
  its evaluated VALUE are both observable; its `external_payload` derives the content hash from
  the `fn` param exactly as the shipping backends do, which is what makes two colliding callables
  one node. `colliding_sessions()` builds two sessions of the same SHAPE and different content, so
  the foreign array's id addresses a real node here and which store answered is visible.
  `scaler` (one factory, two closures), `plus_one`/`times_hundred` (distinct `__name__`s),
  `third_party_verb` (the M23 `descriptor=`+`form=` seam, so it records on any backend) and
  `varied_vector` (three numpy-backed universes) are the O/E operands.
* `m60_lib/__init__.py` + `m60_lib/sub.py` + `m60_libx.py` — real importable modules, not stubs:
  `capture()` classifies a frame by its module `__name__`, so `m60_lib` and `m60_lib.sub` are what
  a registration must cover and the sibling `m60_libx` is what it must not. `m60_lib.q` and
  `m60_libx.q` are also O's same-name-different-module pair.

## Traceability (contract line → test → what it witnesses)

| Line | Test | Mechanism witness |
|---|---|---|
| X1 (instrument) | `test_session_guard.py::test_the_instrument_finds_no_array_taking_method_without_a_leg` | `inspect.getmembers(Session)` by ANNOTATION against the legs below |
| X1 (premise) | `test_session_guard.py::test_a_foreign_arrays_id_addresses_a_real_node_of_the_refusing_session` | `beta.node_id == alpha.node_id`, and each session's own form for that id |
| X1 | `test_session_guard.py::test_every_entry_refuses_a_foreign_array_and_records_nothing` | per method: the exception TYPE, `node_count()` unmoved, and `record_op`'s message on the recording half |
| X2 (instrument) | `test_session_guard.py::test_the_instrument_finds_no_module_verb_taking_a_session_without_a_leg` | `graphed.__all__` walked for a `session` parameter |
| X2 | `test_session_guard.py::test_a_module_verb_refuses_a_foreign_array_and_records_nothing` | `compile_ir`'s `TypeError` and `node_count()` unmoved |
| X3 | `test_session_guard.py::test_own_session_calls_record_and_read_exactly_as_today` | every read's answer, the four `record_*` entries' forms and node count, two builds' IR bytes |
| O1 | `test_callable_identity.py::test_two_distinct_callables_are_two_nodes_that_evaluate_to_their_own_results` | 3 pairs × {`map`, `apply`}: node ids, node count, each materialized value, distinct `fn` params |
| O1 (gufunc) | `test_callable_identity.py::test_two_distinct_callables_are_two_gufunc_nodes` | the same, through `graphed.numpy.apply_gufunc` |
| O1 (durable) | `test_callable_identity.py::test_the_durable_path_ships_each_callable_its_own_evaluator` | `compile_ir` → `pickle` → `evaluate_ir`: two distinct `external_key`s and two distinct results |
| O1 (ordinal) | `test_callable_identity.py::test_the_second_object_of_a_derived_name_records_that_name_with_an_ordinal` | the literal `fn` params `["q", "q#1"]` |
| O2 | `test_callable_identity.py::test_one_callable_object_recorded_twice_is_one_node_and_apply_interns_with_map` | one node id across `map`/`map`/`apply`, node count, the materialized value |
| O3 | `test_callable_identity.py::test_a_program_without_name_collisions_records_the_bare_names_and_the_same_ir_twice` | the literal `fn` params and the IR bytes of two independent builds |
| O3 (collisions) | `test_callable_identity.py::test_two_builds_of_a_colliding_program_agree_byte_for_byte` | one distinct byte string across two builds of the colliding program |
| O4 | `test_callable_identity.py::test_an_explicit_name_is_the_callers_identity_declaration` | equal `name=` interns two distinct callables, distinct `name=` does not, the literal params, and `name=` in each recording verb's docstring |
| V1 | `test_internal_frames.py::test_a_registered_prefix_matches_whole_dotted_components` | the file / line / function / sub-expression of an op the library recorded, for `m60_lib`, `m60_lib.sub` and `m60_libx` |
| V2 | `test_internal_frames.py::test_registration_is_idempotent_and_survives_concurrent_registrars` | a repeated registration's provenance, then 8 threads registering and recording concurrently |
| V2 (control) | `test_internal_frames.py::test_an_unregistered_module_and_the_built_in_graphed_rule_keep_their_own_lines` | `m60_libx`'s own frame, and the test's own frame for a directly recorded op |
| V3 | `test_internal_frames.py::test_a_registered_librarys_stage_error_points_at_the_same_user_line` | `StageError.user_frame`'s file / line / function and `cause_type` |
| E1 | `test_expand_public.py::test_expand_is_public_and_is_the_systematics_verb` | object identity with `graphed.systematics.varied.expand`, and `__all__` membership |
| E1 (varied) | `test_expand_public.py::test_a_wrapped_third_party_verb_answers_a_varied_over_a_varied_operand` | the answer's type, its labels, one distinct node per universe, the node count |
| E1 (unvaried) | `test_expand_public.py::test_an_unvaried_call_passes_straight_through_with_the_bare_calls_ir` | the bare call's node count, materialized value and IR bytes against the wrapped call's |
| E1 (surface) | `test_expand_public.py::test_expand_stays_outside_the_array_consuming_verb_surface` | `expand`'s annotations and its absence from `VERB_DISPOSITIONS` |

## Non-vacuity — what happens on a pre-m60 tree

The suite COLLECTS with zero errors (37 tests; every m60-new outcome is reached inside a test body)
and gives the same verdict on two consecutive runs. 27 legs FAIL, each for its own reason:

* X1 ×8 — `Failed: DID NOT RAISE`. Only `record_op` checks its inputs today, so the three other
  `record_*` entries splice the foreign node in and the five reading entries answer with this
  session's node of the same id. `record_join`'s refusal needs a joinable pair to be visible at
  all: with an ill-typed pair it raises for a backend reason instead.
* X2 — the same, for `compile_ir`.
* O1 ×10 — `assert 1 != 1` on the two node ids: two distinct callables deriving one name intern to
  ONE node, and `session.node_count()` is 2 where the contract wants 3. The durable leg fails one
  assertion earlier, on `len(set(keys)) == 2`: one node ships one payload key.
* O1 (ordinal) — `assert ['q'] == ['q', 'q#1']`: one node, one `fn` param.
* O4 — `assert Array.map.__doc__ is not None`. The interning half runs first and passes, so what
  fails is only the half the contract moves.
* V1/V2/V3 — `AttributeError: module 'graphed.provenance' has no attribute 'register_internal'`.
* E1 ×3 — `AttributeError: module 'graphed' has no attribute 'expand'`. The unvaried leg's BARE
  call runs first and passes, so the failure is the public name, not the behaviour.

10 legs PASS on a pre-m60 tree — the declared controls, each a live instrument showing the harness
reaches the recorder, the provenance capture and the verb table at all:

* X1/X2's two instruments: today's discovered method and verb sets, which red when a member grows
  without a leg.
* X1's premise leg: the ids really do collide, which is what makes today's answers silently wrong.
* X1's `record_op` leg: the one refusal that already exists, message included, which X3 keeps.
* X3: every own-session read and record, and two builds' IR bytes.
* O2, O3 ×2: the interning and byte-identity properties the ordinal must not disturb.
* V2's control: an unregistered module's own line and the built-in `graphed*` rule.
* E1's surface leg: `expand` joining `__all__` adds no m48 disposition obligation.

## What the frozen expectations were measured against

No assertion here waits on the implementation to be executed for the first time: every leg sitting
behind a first failing one was run by simulation on a pre-m60 tree, under a monkeypatching
stand-in applied as a pytest plugin outside this tree (all 37 legs green).

* The guard was simulated on all nine `Session` entries plus `compile_ir`; every refused call's
  `node_count()` and every own-session answer in X3 was measured under it.
* The ordinal was simulated on `Array.map`, `graphed.apply` and `graphed.numpy.apply_gufunc`: the
  three pairs' node ids and materialized values, the `["q", "q#1"]` params, and the durable path's
  two `external_key`s and two results were all measured there.
* `register_internal` was simulated as a component-matching skip set inside `capture`, and V1's
  file/line/function/source, V2's eight concurrent registrars and V3's `StageError.user_frame`
  were measured against it.
* `graphed.expand` was simulated as the same object re-exported; E1's `Varied` labels, per-universe
  node ids and the unvaried call's IR bytes were measured there. Adding `expand` to `__all__` was
  separately run against the older frozen enumerations that walk it (`frontend/m48`,
  `awkward/m48`): 112 tests, unchanged.

## Where a contract line is pinned by its nearest observable

* **X1's method set comes from ANNOTATIONS, X2's from a parameter NAME.** Every `Array` parameter
  of a public `Session` method annotates `Array`, `Sequence[Array]` or `*outputs: Array`, while
  their names are six different spellings — so the annotation walk is the reliable one there.
  `compile_ir`'s array parameter is `*outputs: Any`, so an annotation walk finds nothing to guard
  and X2 keys on the `session` parameter instead.
* **X1 covers `serialized_ir`, which the decomposition's list of reading entries omits.** It takes
  `*outputs: Array`, the instrument finds it, and R25.3 says EVERY method that takes one; it is
  pinned as a reading entry (`TypeError`).
* **`GraphedTypeError` is not a `TypeError` subclass**, so `pytest.raises(TypeError)` on the
  reading entries and `pytest.raises(GraphedTypeError)` on the recording ones separate the two
  halves of the contract by themselves.
* **O4's docstring half is pinned as the substring `name=`.** A frozen test cannot assert prose;
  what it can assert is that each recording verb documents the keyword at all, which no verb does
  today (`Array.map` has no docstring).
* **V2's thread-safety is pinned as eight concurrent registrars agreeing.** Nothing observable
  distinguishes a locked registry from a lucky one; what a frozen test can pin is that concurrent
  registration and recording neither raises nor moves any thread's user line.
* **The V legs are not process-isolated.** Registration is process-global and today's module offers
  no inverse, so every V leg lives in one module and `"m60_lib"` is the only prefix this suite ever
  registers; by the dotted-component rule under test it cannot reach `m60_libx`, no other m60 leg
  records an op from inside `m60_lib`, and the frozen frontend tree runs one process per milestone
  dir (`scripts/run-tests.sh`), so nothing outside this dir can observe the registration.
* **The ordinal separator is pinned only where the decomposition states it** — the literal pair
  `["q", "q#1"]`. Every other O leg asserts distinctness, not spelling.
