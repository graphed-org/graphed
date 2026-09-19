# awkward/m58 — projection honours what sources and Externals declare (traceability)

Authority: `graphed-workdir/m58-decomposition.md` (contract lines H1–H5, E1–E3, one frozen property
each) under root-prompt rule **R25.1**. Two decisions move:

* **integ-m58-H** — every driver that feeds `PartitionedSource.read_partition` asks the source DATA
  object for `projected_columns(outputs)` when it has that attribute (found with `getattr`; the
  `runtime_checkable` Protocol gains NO member), driver-side, once per driver call, with all the
  outputs in the caller's order, and ships the answer unsorted and undeduped in the plan.
* **integ-m58-E** — in projection replay the value standing in for an External's output is a
  typetracer of the node's RECORDED form, carrying the backend's behavior, never the first input.

Run: `python -m pytest tests/frozen/awkward/m58 -q` (its own process, per the awkward
per-milestone split). The `m58_` helper prefix is load-bearing under prepend import mode.

## Fixture — `m58_declaration_fixtures.py`

* `DATA` — four events of `{x: var*float64, y: var*float64, z: float64}`; every value is dyadic, so
  every asserted sum, product and written column is exact.
* `PlainSource` — a `PartitionedSource` over `DATA` that RECORDS the `columns` argument of every
  `read_partition` call (raw, so a list would not compare equal to a tuple). Its whole-dataset
  loader asserts if a plan ever calls it.
* `DeclaringSource` — `PlainSource` plus `projected_columns`, counting its calls, keeping the exact
  `outputs` tuple it was handed, and returning a **list** so the driver is the one that makes the
  answer a tuple. `max_calls` turns any further call into an `AssertionError`. A pickled copy is the
  reader a worker gets: the answer and the call counter travel with it, the observations of the
  building process (`outputs`, `seen`) do not — so a worker copy starts with an empty read log and a
  counter already at its limit, and the live session `Array`s the hook was handed stay here.
* `DECLARED = ("z", "y", "x", "x")` — unsorted, duplicated, and naming a column (`z`) that neither
  the aggregate nor the parquet program reads, so sorting, deduping, reordering or ignoring the
  answer is visible at `read_partition`. It is a superset of every program's needs, so the plans
  still evaluate and the value assertions hold with and without the hook.
* `AGGREGATE_COLUMNS` / `PARQUET_COLUMNS` / `VARIED_COLUMNS` — the three drivers' own answers on
  this dataset, which H2 and H5 pin literally.
* `aggregate_over` / `varied_record` and the module-level `counts` / `add_counts` / `no_counts`
  (picklable, so the aggregate closure ships through `pickle` and a process pool alike).
* `declared_external(session, inputs, form, tag)` — the M23 seam
  (`record_external(descriptor=, form=)`), whose evaluator `never_runs` asserts if projection ever
  calls it. `NESTED` is one list level DEEPER than `DATA.x`; `PAIRS` is a record named `m58pair`,
  which only `BEHAVIOR`'s `PairArray.total` property understands and which global `ak.behavior`
  never sees.

## Traceability (contract line → test → what it witnesses)

| Line | Test | Mechanism witness |
|---|---|---|
| H1 (aggregate) | `test_read_list.py::test_the_aggregate_driver_ships_the_sources_declared_read_list` | hook call count, `outputs[i] is out_i`, `plan.process.columns`, both recorded `read_partition` column tuples |
| H1 (parquet write) | `test_read_list.py::test_the_parquet_write_driver_ships_the_sources_declared_read_list` | hook count + `outputs[0] is output`, recorded columns, written payload |
| H1 (varied write) | `test_read_list.py::test_the_varied_write_driver_ships_the_sources_declared_read_list` | hook count, the outputs are this session's `Array`s, recorded columns, every universe read back |
| H2 | `test_read_list.py::test_a_source_without_the_hook_keeps_each_drivers_own_read_list` | all three drivers' own column tuples, literally |
| H2 (protocol) | `test_read_list.py::test_the_partitioned_source_protocol_gains_no_member` | `isinstance` both ways, plus a hook-only object that is NOT a `PartitionedSource` |
| H3 | `test_plan_boundary.py::test_the_built_plan_carries_the_answer_and_no_worker_asks_again` | call counter across build + run, then a `pickle` round-trip whose source raises on a second call |
| H3 (fixture control) | `test_plan_boundary.py::test_a_pickled_declaring_source_keeps_its_counter_and_drops_its_witness_state` | what `pickle` does to the source alone: the counter and answer arrive, the witness state does not, a further call raises, and the copy records its own first read |
| H4 | `test_plan_boundary.py::test_two_builds_agree_byte_for_byte_and_keep_the_sources_own_order` | `bytes(plan.process.ir)` equality, then the declared order and duplicate at `read_partition` |
| H5 | `test_plan_boundary.py::test_read_columns_by_label_is_the_same_with_and_without_a_hook` | the per-label answers agree and the hook's counter stays 0 |
| E1 | `test_external_stand_in.py::test_an_external_declared_one_level_deeper_projects_on_its_own_form` | `gak.num(root.x, axis=2)` is ill-typed (so the op needs the declaration), then both projection reports |
| E2 | `test_external_stand_in.py::test_a_shape_preserving_external_reports_exactly_what_it_reported_before` | the column report AND the buffer report (`x: DATA`, `y: OFFSETS`) pinned literally |
| E3 | `test_external_stand_in.py::test_the_stand_in_carries_the_session_backends_behavior` | the declared form's record name, the recorded property form, both projection reports |

## Non-vacuity — what happens on a pre-m58 tree

The suite COLLECTS with zero errors (12 tests; every m58-new outcome is reached inside a test body)
and gives the same verdict on two consecutive runs. Seven legs FAIL, each for its own reason:

* H1 ×3, H3 — `assert 0 == 1` on `source.calls`: no driver consults the hook today, so the counter
  never moves.
* H4 — `assert ('x', 'y') == ('z', 'y', 'x', 'x')`. The byte-identity half passes first and is
  asserted first, so the failure is exactly the half that needs the hook.
* E1 — `numpy.exceptions.AxisError: axis=2 exceeds the depth of this array (2)`: the stand-in is
  `inputs[0]`, which is one level shallower than the node's declared form.
* E3 — `AttributeError: no field named 'total'`: the stand-in is `inputs[0]` (`events.z`), which
  carries neither the `m58pair` name nor the behavior property.

Five legs PASS on a pre-m58 tree — the declared controls, each a live instrument showing the harness
reaches the drivers, the projection and the fixture at all:

* H2's two legs: today's read lists per driver, and the `isinstance` checks a Protocol member would
  break for every source that has no hook.
* H5: `read_columns_by_label` is syntactic and has no source object; its answer and the hook's
  zero call count must both survive.
* E2: a shape-preserving External's reports, pinned literally, so a stand-in change that also moves
  the unchanged case is caught.
* the H3 fixture control: the source's own `pickle` contract, which H3's shipped-plan legs stand on
  and which needs no hook, so it is a pre-m58 instrument for the boundary H3 asserts across.

## What the frozen expectations were measured against

No assertion here waits on the implementation to be executed for the first time: every leg sitting
behind a first failing one was measured on a pre-m58 tree by simulation.

* The three H drivers were run with their plans' `columns` forced to `DECLARED` — `dataclasses.replace`
  on the aggregate plan, the write drivers' column seam substituted out of tree — and the output
  tuple each driver computes handed to `projected_columns` by hand. All three still produce the
  values asserted here, both recorded `read_partition` tuples are `DECLARED`, and the outputs are
  the caller's own objects, so no leg fails post-implementation for a fixture reason.
* H3's shipped-plan legs were run on such a forced plan: it pickles, the copy's `columns` is
  `DECLARED`, it reduces to `AGGREGATE_VALUE`, its reader's counter reads 1 and its read log holds
  the two reads of that run and no others.
* E1/E3's reports — column and buffer alike — were measured against a replay whose `on_external`
  returns the recorded form's typetracer.

Note that `AwkwardBackend.eval_stage` re-applies the backend's behavior to every operand, so E3
pins the observable the contract states — the property RESOLVES during projection — not the route
the behavior takes to the stand-in; no projection-visible difference distinguishes the two.
