# awkward/m59 — awkward-idiom parity at the frontend (traceability)

Authority: `graphed-workdir/m59-decomposition.md` (contract lines I1–I6, S1–S5, M1–M4, one frozen
property each) under root-prompt rule **R25.2**. Three frontend decisions move, all in
`graphed/array.py` plus each backend's evaluation/form inference:

* **integ-m59-I** — `Array.__getitem__` accepts a TUPLE key of int-field `slice`s, `int`s, `None`
  and at most one `Ellipsis`, provided the partitioned axis is left whole (first member
  `slice(None)`, or a leading `Ellipsis`). Such an op is per-row and fusible, never a boundary.
* **integ-m59-S** — a numpy scalar operand keeps its dtype AND its exact value, instead of
  `float(value)`; Python operands record exactly as before, and `ParamValue`, the Rust store and
  the frontend's import set are unchanged.
* **integ-m59-M** — a behavior method may return a NESTED tuple of arrays and gets the same nesting
  of `graphed.Array`s back.

Run: `python -m pytest tests/frozen/awkward/m59 -q` (its own process, per the awkward
per-milestone split). The numpy-backend half of I6 is `tests/frozen/numpy/m59`. The `m59_` helper
prefix is load-bearing under prepend import mode.

## Fixture — `m59_idiom_fixtures.py`

* `DATA` — four events of `{x: var * float64, y: var * float64}` with NO empty list, so `[:, 0]`
  and `[:, -1]` are defined on every row and an I1 failure is about the key, never raggedness.
  Every value is dyadic, so every asserted product, sum and widened float32 is exact.
* `RAGGED` — three rows, the middle one empty: the run-time index error I5 needs. A typetracer
  cannot see it, so the op records and fails only when real data arrives.
* `ACCEPTED` / `CONSUMES_AXIS0` / `MALFORMED` — the key sets, by their spelling.
* `ARRAYS` × `SCALARS` — S1's matrix: {bool, int64, float64} arrays × {`np.uint64`, `np.int32`,
  `np.float32`, `np.bool_`} scalars. `PY_SCALARS` is S3's side.
* `PlainSource` + `partitioned` + `rows`/`merge`/`nothing`/`concatenated` — the I2 partitioned run:
  one aggregate entry per partition carrying that partition's own TYPE and rows, so a partitioned
  run is compared to an unpartitioned one on structure as well as values.
* `PairArray` + `BEHAVIOR` + `PAIRS` — a behavior class the backend alone knows, with `nested`
  (`(a, (b, c))` — `metric_table(return_combinations=True)`'s shape), `flat`, `one`, and `leafy`
  (the same nesting with a non-array leaf).
* `eager_form` / `tracer` / `eager_pairs` / `tracer_pairs` — eager awkward as the ORACLE: the same
  expression on a typetracer of the same data is what a recorded form must equal, and on the real
  array is what execution must equal.

## Traceability (contract line → test → what it witnesses)

| Line | Test | Mechanism witness |
|---|---|---|
| I1 | `test_multi_axis_keys.py::test_a_multi_axis_key_records_and_executes_as_eager_awkward_does` | `session.form(out).describe()` against the typetracer oracle, executed values against the real array — 7 keys × (jagged field, record of jagged fields) |
| I2 (flag) | same test's last assertion | the node's `kind` read back from the serialized IR is `op`, not `reduction` |
| I2 (run) | `test_multi_axis_keys.py::test_a_partitioned_run_of_a_multi_axis_key_equals_the_unpartitioned_one` | one `read_partition` per partition, two partitions vs one, per-partition type sets, concatenated rows against eager |
| I3 (set) | `test_key_refusals.py::test_a_refused_tuple_key_records_nothing` | `TypeError` **and** `node_count()` unmoved, for the axis-0 group and the malformed group |
| I3 (Array member) | `test_key_refusals.py::test_an_array_member_inside_a_tuple_key_is_refused` | same, for the non-goal `a[:, mask]` |
| I3 (message) | `test_key_refusals.py::test_consuming_the_partitioned_axis_is_refused_by_naming_the_chained_spelling` | the message carries a two-subscript spelling (`][`), which is the observable form of "naming the chained spelling `a[1:3][:, 0]`" |
| I4 | `test_multi_axis_keys.py::test_equal_keys_intern_and_two_builds_agree_byte_for_byte` | node-id equality for one key, pairwise distinctness across seven, `serialized_ir` bytes across two sessions |
| I5 | `test_key_refusals.py::test_an_index_past_an_empty_row_surfaces_as_the_executors_stage_error` | the recorded form first (so the refusal is NOT at record time), then `StageError.cause_type`, the user frame's source text and enclosing function |
| I-control | `test_multi_axis_keys.py::test_every_key_accepted_today_records_the_same_op_params_and_boundary_flag` | (kind, name, params) for `field`, `fields`, `slice`×2, `index`, `getitem`, plus two executed values |
| S1 | `test_scalar_dtype.py::test_a_numpy_scalar_operand_keeps_its_dtype_on_either_side` | recorded form, executed dtype and executed values against eager, for `*`/`+`/`-` on both sides and `>` |
| S1 (ops) | `test_scalar_dtype.py::test_bitwise_operators_and_ufuncs_keep_the_scalar_dtype_too` | `&`/`\|`/`^`/`<<` and `np.add`/`np.subtract`/`np.maximum`, incl. `mask * np.uint64(1 << 3)` |
| S2 | `test_scalar_dtype.py::test_a_scalar_value_wider_than_a_float_round_trips_exactly` | `np.uint64(2**63)` and `np.uint64(2**64 - 1)` as exact ints, and `np.float32(0.1)` widened to `0.10000000149011612` |
| S3 | `test_scalar_dtype.py::test_python_scalars_record_exactly_as_before` | the literal `params` of `mul` for int/float/bool, the `side` param, and `serialized_ir` bytes against a fresh session |
| S4 | `test_scalar_dtype.py::test_dtype_and_value_together_decide_a_scalar_operands_node` | five spellings of "one" give five nodes; equal dtype+value gives one; a different value gives another |
| S5 | `test_scalar_dtype.py::test_the_frontend_still_imports_neither_numpy_nor_awkward` | a FRESH interpreter: `sys.modules` after `import graphed`, then three distinct nodes through a stub backend and every IR param value a `ParamValue` |
| M1 | `test_nested_method_outputs.py::test_a_nested_tuple_method_returns_the_same_nesting_of_arrays` | the unpacking `metric, (left, right)`, each leaf's form against the typetracer oracle, each leaf's values against eager |
| M2 | `test_nested_method_outputs.py::test_single_and_flat_returns_record_exactly_as_before` | the literal `method` params (incl. the flat tuple's `index`), values, and `serialized_ir` bytes across two sessions |
| M3 | `test_nested_method_outputs.py::test_a_non_array_leaf_is_refused_while_the_same_nesting_of_arrays_records` | the existing refusal's shape (`leafy():` … `cannot be recorded`), then the SAME nesting with array leaves recording |
| M4 | `test_nested_method_outputs.py::test_the_same_nested_call_interns_and_distinct_leaves_are_distinct_nodes` | the three leaf ids of two identical calls, and their pairwise distinctness |

## Non-vacuity — what happens on a pre-m59 tree

The suite COLLECTS with zero errors (48 tests; every m59-new outcome is reached inside a test body)
and gives the same verdict on two consecutive runs. 31 legs FAIL, each for its own reason:

* I1 ×14, I2's two, I4 — `TypeError: unsupported index (…); use an Array mask/index, a slice, an
  int, or a field name` from `Array.__getitem__`: no tuple key is accepted at all today.
* I3's message leg — `assert '][' in 'unsupported index (slice(1, 3, None), 0); …'`: the catch-all
  refusal does not tell the user the spelling that works.
* I5 — the same `TypeError`, at RECORD time, which is exactly what the contract moves to run time.
* S1 ×6 — `assert '## * var * float64' == '## * var * uint64'` (and `int32`, `float32`, `bool`):
  `float(value)` erases the dtype wherever float64 does not absorb it anyway.
* S1's bitwise leg — `ufunc 'bitwise_and' not supported for the input types`: a `bool` array and a
  float-collapsed `np.bool_` is not even a legal awkward expression.
* S2 — the same form mismatch; the value half needs a param the store can hold (an `int` param is
  i64, so `2**63` and above cannot travel as one).
* S4 — `{'uint64': 1, 'int32': 1, 'float32': 1, 'python_int': 2, 'python_float': 1}`: four of the
  five spellings of "one" intern to the SAME node today.
* S5 — the subprocess exits non-zero at the three-distinct-nodes assertion. Its purity assertion
  runs FIRST and passes, so the subprocess is a live instrument, not a dead one.
* M1, M4 and M3's second half — `nested() returned tuple, which is not an awkward array or a tuple
  of awkward arrays, so it cannot be recorded`: a nested tuple is refused wholesale today.

17 legs PASS on a pre-m59 tree — the declared controls, each a live instrument showing the harness
reaches the recorder, the backend and the executor at all:

* the I-control: today's (kind, name, params, boundary flag) for every key `__getitem__` accepts.
* I3's seven refusal params and the `Array`-member leg: today's catch-all already refuses them, and
  they must stay refused — nothing may be recorded on the way out.
* S1's six dtype-ABSORBING rows (every `float64` row, `int64`×`uint64`, `int64`×`float32`): eager
  awkward answers float64 there whatever the frontend does, so they pin the values and the operand
  `side` rather than the dtype.
* S3 and M2: the two "exactly as before" regression pins.

## What the frozen expectations were measured against

No assertion here waits on the implementation to be executed for the first time: every leg sitting
behind a first failing one was run by simulation on a pre-m59 tree (146 legs, all green).

* Every I1 form/value pair was evaluated eagerly, on the typetracer and the real array, for all
  seven keys over both the jagged field and the record — and `AwkwardForm(...).describe()` renders
  the same string the oracle does, so the comparison is well-posed.
* I2's whole fixture — `PlainSource`, `partitioned`, `aggregate_plan`, `SequentialRunner`,
  `rows`/`merge`/`nothing`/`concatenated` — was run at one and two partitions with the ALREADY
  SUPPORTED spelling of the same selection (`x[gak.local_index(x, axis=1) < 2]` is `x[:, :2]`):
  fusible node, one read per partition, and the concatenation equals eager both ways.
* I4's interning and cross-session byte identity were run on today's `slice` keys.
* I5's `StageError` was provoked with the m6 spelling of the same failure (the index inside a
  `map`): `cause_type == "IndexError"`, the user frame names the key text and the enclosing
  function. Its recorded-form oracle is `## * float64`.
* Every S leg was run through `graphed.awkward._ops.apply` with the numpy scalar left in the
  params — the eval path the backend takes once the dtype survives — for the whole matrix, both
  sides, the bitwise/ufunc set and the wide values.
* M's leaf forms and values were read off eager and typetracer `nested()` calls
  (`[2.0, 8.0]`, `[3.0, 9.0]`, `[3.0, 12.0]`, each `## * float64`).

## Where a contract line is pinned by its nearest observable

* **I3's message.** "Naming the chained spelling" is pinned as a two-subscript spelling (`][`) in
  the message, and only for the spec's own example `a[1:3, 0]`; the int- and `None`-first members
  are pinned as plain refusals, because `a[0]` and `a[None]` have no chained spelling to name.
* **I3's exception type.** `TypeError`, as the contract says — not `GraphedTypeError`, which is not
  a `TypeError` subclass. Key validation happens before any backend call, as it does today.
* **S1's comparison legs are array-side only.** `np.uint64(8) > ak.Array(...)` raises in EAGER
  awkward itself (`could not find singular backend for ndarray, ListOffsetArray`), so there is no
  eager answer for a numpy scalar on the left of a comparison to be at parity with.
* **S1's subtraction legs skip the bool array.** numpy forbids boolean subtract; `*`, `+` and `>`
  cover that row, and the asymmetric reflected path is pinned on int64/float64 and by
  `np.subtract(np.int32(2), a)`.
* **S5 does not pin how the dtype is spelled.** The contract's observable is that the value travels
  as a `ParamValue` and that the frontend imports neither backend, so the test pins the `ParamValue`
  closure and the distinctness of the nodes, not a param name or a dtype spelling.
