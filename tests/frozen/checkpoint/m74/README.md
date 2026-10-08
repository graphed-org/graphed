# m74 frozen suite — `graphed.checkpoint.resumable`

Plan: `graphed-workdir/lanes/ckpt-resume/plan.md` (r14b) §2, §4.1 T1–T10; interface addendum (`resumable(plan,
store, *, storage_options=None, codec=None, salt="", accept_environment=False)`). **Frozen — read-only after
`freeze-m74`.** On main every test fails at import: four modules import `resumable`/`check_resumable`/
`StoreUnavailable`/`EnvironmentChanged` from `graphed.checkpoint`, and every cross-interpreter test fails on
the same `ImportError` inside its subprocess.

## Files

| File | Role |
|---|---|
| `m74_helpers.py` | importable processes, codecs, spies, plan builders, `run_script` (subprocess runner), fake dist-info, store readers |
| `m74_xleg.py` | T2's cross-interpreter legs; each leg's keyed objects are `__main__` objects of this script |
| `m74_env.py` | T3's runs, one interpreter per run |
| `test_m74_resume.py` | T1, T5, T6, T7, T8, T9, T10 |
| `test_m74_key.py` | T2 in process |
| `test_m74_key_interpreters.py` | T2 cross-interpreter (i)–(xv) |
| `test_m74_environment.py` | T3 |
| `test_m74_refusals.py` | T4 |

"Done" is `len(store.completed())` (§4). Subprocess legs set `PYTHONHASHSEED` to fixed distinct values
(first run `1`, later runs `2`, `3`, …).

## Traceability

| Test | Plan clause | Wrong implementation it fails |
|---|---|---|
| `test_resume_runs_only_the_tasks_not_done` | T1, §2.4, §2.7 | no reuse; reuse without recording done tasks |
| `test_rebuild_salt_codec_process_and_partition` | T2 rebuild/salt/codec/bins/one partition | key ignoring salt, codec, process state or partition |
| `test_wraps_closures_differing_in_a_captured_value` | T2 wraps closures | key by `__wrapped__`/qualname |
| `test_closures_differing_in_a_default` | T2 `k=k` | function defaults dropped from the key |
| `test_closures_differing_in_an_attribute` | T2 function attribute | function `__dict__` dropped |
| `test_type_classes_differing_in_a_class_attribute` / `_in_a_base` / `_whose_call_closes_over_different_values` | T2 `type()` classes | class keyed by name or public data only |
| `test_lambdas_differing_in_body` | T2 lambdas | lambdas keyed by name |
| `test_class_holding_a_module_is_accepted_and_reused` | T2 module attribute | module pickled by value / refused |
| `test_enum_member_is_accepted_and_reused` | T2 enum | tracker ids left in the key |
| `test_class_holding_a_set_of_its_own_instances_is_accepted_and_reused` | T2 recursive set | refusing it (recursion) |
| `test_importable_class_with_class_level_lock_and_cache` | T2 importable lock/cache | importable classes keyed by value (refused or moved by the cache) |
| `test_main_defs_whose_try_covers_different_statements` | T2 try coverage (fails only on ≤ 3.13) | `co_exceptiontable` blanked as a location |
| `test_runtime_module_global_attribute_change[False/True]` | T2 `types.ModuleType` global | spec-less module pickled by name when in `sys.modules` |
| `test_main_codec_classes_with_different_encode_bodies` | T2 codec in the key | codec left out of the key |
| `test_frozenset_dataclass_process` | T2 (i); constraint: two fixed seeds, plain-pickle digests differ | unsorted set bytes |
| `test_aggregate_plan_from_an_importable_builder` | T2 (ii) | per-process state in a graphed plan's key |
| `test_aggregate_plan_under_two_cell_filenames` | T2 (iii); cell exec'd into `__main__.__dict__` | `co_filename`/frames in the key |
| `test_shuffle_plan_main_reduce_under_two_cells[class/def]` | T2 (iv); constraint: both variants | V2 keyed by `plan.task_id` (opaque cloudpickle bytes) |
| `test_shuffle_plan_loaded_from_pickle_in_another_interpreter` | T2 (v), observable form per constraint | idents differing between a built and a loaded plan |
| `test_frozen_dataclass_with_defaults` | T2 (vi) | dataclass doc/default reprs or set order in the key |
| `test_main_reduce_reading_a_module_global` | T2 (vii) | line tables/first line in the key; `BINS` or body left out |
| `test_generic_process_class` | T2 (viii) | `TypeVar` tracker ids in the key |
| `test_cloudpickled_plan_loaded_twice` | T2 (ix) | memo/identity-dependent bytes |
| `test_copyreg_reducer_registered_after_import` | T2 (x) | `copyreg` table snapshotted at import |
| `test_abc_registry_and_weakset_made_in_another_order` | T2 (xi); the second run makes classes and set members in reverse | unsorted ABC registry / `WeakSet` |
| `test_main_result_class_rebuilt_in_one_interpreter` | T2 (xii) | `__slotnames__` cache in the key |
| `test_mixin_edit_reaches_an_aggregate_key[global/ref/callable]` | T2 (xiii) global `ak.behavior`, `backend="m:f"`, `backend=m.f` | `behaviors` not folded; `_PartitionReduce.checkpoint_resolve` dropped; only str refs resolved |
| `test_mixin_edit_reaches_shuffle_stages[before/after]` | T2 (xiii) shuffle, mixin before/after `repartition` | `_MapWrite` declaration dropped (before layout); unchained V2 idents. The after layout does not guard the `_Gather` declaration (map and gather share one factory) |
| `test_mixin_edit_rewrites_parquet_parts[parquet/varied]` | T2 (xiii) `to_parquet(behavior="m:a")`, plain and varied with `select=`, one fixed destination | `_WritePart` / `_VariedWritePart` declaration dropped |
| `test_importable_object_with_a_uuid_reduce` | T2 (xiv); `__qualname__` set on the instance | no by-name rule before library reduces |
| `test_upstream_edit_reaches_every_downstream_stage[gather/reader/join]` | T2 (xv) | unchained V2 idents (gather); idents folding only `inputs[0]` (join) |
| `test_same_environment_resumes_and_an_added_distribution_refuses` | T3 same env / added | no environment record |
| `test_changed_and_removed_distributions_are_named` | T3 `1.0 -> 2.0`, removed | comparing names only |
| `test_accepting_alternating_environments` | T3 accept ×5, plain after each, A after accepting B refuses | accepting a match with any record, not the greatest `n` |
| `test_the_copy_python_imports_is_the_one_fingerprinted` | T3 2.0 ahead of 1.0, then 3.0 | last copy in `distributions()` order |
| `test_two_first_drivers_in_two_environments` | T3 two records at `n = 1`; plain under A and B both refuse | comparing against one record in force |
| `test_a_cloudpickle_version_change_refuses` | T3 cloudpickle | cloudpickle outside the fingerprint |
| `test_a_new_salt_reuses_nothing_and_does_not_refuse` | T3 new salt | salt-blind records |
| `test_environment_changed_survives_pickle` | T3 pickle | unpicklable exception |
| `test_serial_runners_check_the_environment[run_resumable/run_shuffle_resumable]` | T3 serial runners; the fill is interrupted after 2 tasks so the refusal is checked before any process call | check missing, or after the task loop |
| `test_serial_runners_refuse_a_store_without_record_environment` | T3 six-method store | silently skipping the check |
| `test_plan_with_next_tasks` … `test_already_resumable_plan` | T4 refusals; root not created; `check_resumable` raises the same | refusal missing, or after store I/O / pickling (`Served.__reduce__` spy) |
| `test_process_reading_a_main_cache_def` | T4 `__main__` cache def | no `__main__` by-name refusal |
| `test_allowed[...]` | T4 allowed: lambda, `aggregate_plan` with services, `shuffle_plan`, `to_parquet(compute=False)` over a service External | over-refusal; plan not keyed |
| `test_inner_process_sees_the_original_partitions` | T5 `Partition` type/equality incl. blind | rebuilding `Partition` from fewer fields |
| `test_bind_and_resolve_services_reach_the_inner_process` | T5 forwarding | hooks not forwarded |
| `test_a_blob_lost_after_resumable_returns_is_recomputed[deleted/corrupted]` | T5 | serving or failing on a missing/unverifiable blob |
| `test_three_live_threads_write_three_journals[dir/memory/file]` | T6; `Barrier(3, timeout=10)` | one writer per process; a lock around `inner` (fails on the barrier timeout) |
| `test_durable_plan_becomes_a_runtime_plan_keyed_by_task_id` | T7 `DurablePlan` | no conversion; keys not journaled |
| `test_shuffle_plan_interrupted_resumes_with_identical_gather_payloads` | T7 `DurablePlanV2` | payload changed by the stage wrapper; no V2 resume |
| `test_store_unavailable_wraps_store_calls_only` | T8 | a broad `except` turning inner errors into `StoreUnavailable` |
| `test_resumable_decodes_nothing_and_workers_decode` | T9 | decoding stored partials on the driver |
| `test_run_resumable_folds_partials_as_they_come` | T10 order, 1e16 bits, peak live partials ≤ 3 | holding every partial (main: T+2) |
