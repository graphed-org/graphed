# m48 frozen suite — `graphed` / `tests/frozen/frontend` (awkward-free)

Milestone m48 (`vary` frontend + weight path). Authority: `systematics-vary-plan.md` r33.
This tree holds every m48 anchor that needs neither `graphed.awkward` nor an event context
(§10/m48 partition rule (3)). It is collected WHOLE by `ci.yml`'s required `test-freethreaded`
job under `pytest hypothesis numpy` alone, so no module here may import `graphed.awkward`,
`pyarrow`, `hist` or `pandas` — directly or transitively.

| anchor | plan clause | test |
|---|---|---|
| `vary-m48-F1` | §1.2 label-out-of-identity (sibling-mode scoped) | `test_label_out_of_identity.py::test_no_node_name_or_params_carries_a_label` |
| `vary-m48-F1` | §1.2 rename-invariant `compile_ir` bytes | `test_label_out_of_identity.py::test_renaming_every_label_leaves_the_compiled_ir_byte_identical` |
| `vary-m48-F1` | §5.2a dedup witness (arena delta, one node id) | `test_label_out_of_identity.py::test_structurally_identical_members_intern_to_one_node` |
| `vary-m48-F1` | §7.2 merge-guard SCOPE positive control | `test_label_out_of_identity.py::test_unvaried_optimizer_merged_program_still_compiles_and_runs` |
| `vary-m48-F2` | §3.2 two-process byte-identical compile | `test_variation_determinism.py::test_two_fresh_processes_compile_the_same_bytes` |
| `vary-m48-F2` | §2.2 label order (nominal first, insertion order) | `test_variation_determinism.py::test_label_order_is_nominal_first_then_insertion_order` |
| `vary-m48-F2` | §2.1 stacking order (inherited before new) | `test_variation_determinism.py::test_stacking_puts_inherited_labels_before_new_ones` |
| `vary-m48-F3` | §7.2 `Plan` key set | `test_varied_schema_absence.py::test_plan_schema_is_unchanged_by_a_varied_program` |
| `vary-m48-F3` | §7.2 `ExecResult` key set | `test_varied_schema_absence.py::test_exec_result_schema_is_unchanged_by_a_varied_program` |
| `vary-m48-F3` | §7.2 monitor `TaskEvent` key set | `test_varied_schema_absence.py::test_monitor_task_payload_schema_is_unchanged_by_a_varied_program` |
| `vary-m48-F4` | §7.2 seam (alpha): one call, the `CompiledGraph` | `test_varied_aggregate_plan.py::test_hook_fires_exactly_once_with_the_compiled_graph` |
| `vary-m48-F4` | §7.2 seam additivity (m5's own assertion) | `test_varied_aggregate_plan.py::test_the_hooked_plan_runs_to_the_hookless_value` |
| `vary-m48-F4` | §7.2 seam (beta) return channel + §8.2(i) field | `test_varied_aggregate_plan.py::test_the_hook_return_value_is_carried_onto_the_shipped_closure` |
| `vary-m48-F5` | §2.2 `Varied.apply` per universe | `test_varied_apply.py::test_apply_maps_the_function_over_every_universe` |
| `vary-m48-F5` | §2.2 `Varied.apply` idiom of the result | `test_varied_apply.py::test_apply_returns_the_nominal_members_idiom` |
| `vary-m48-F5` | §2.2 `Varied.apply` error contract | `test_varied_apply.py::test_a_function_returning_a_varied_is_refused_with_guidance` |
| `vary-m48-F6` | §2.3a parity gate + §2.3c floor | `test_varied_array_surface.py::test_every_discovered_surface_name_resolves_on_the_varied_class` |
| `vary-m48-F6` | §2.3a class-resolution rule | `test_varied_array_surface.py::test_class_resolution_is_what_makes_the_parity_gate_discriminating` |
| `vary-m48-F6` | §2.3e(4) `Array`-surface floor | `test_varied_array_surface.py::test_surface_floor_names_repartition_as_the_only_refusing_member` |
| `vary-m48-F6` | §2.3a behavioural probe per class | `test_varied_array_surface.py::test_one_behavioural_probe_per_disposition_class` |
| `vary-m48-F6` | §2.2 property dispositions by measurement | `test_varied_array_surface.py::test_properties_are_classified_by_measurement_not_by_a_literal_list` |
| `vary-m48-F6` | §2.2 reserved names + string-getitem control | `test_varied_array_surface.py::test_node_id_and_session_raise_while_the_field_of_that_name_still_reads` |
| `vary-m48-F7` | §2.3b `plain[varied_mask]` | `test_varied_entry_points.py::test_getitem_with_a_varied_mask_returns_a_varied_carrying_the_masks_labels` |
| `vary-m48-F7` | §2.3b `plain.filter(varied_mask)` | `test_varied_entry_points.py::test_filter_with_a_varied_mask_returns_a_varied_carrying_the_masks_labels` |
| `vary-m48-F7` | §2.3b unsupported-index control | `test_varied_entry_points.py::test_the_varied_branch_is_not_a_blanket_except` |
| `vary-m48-F8` | §2.5 unreached-label report | `test_unreached_label_diagnostic.py::test_a_registered_label_that_reaches_no_marked_output_is_reported` |
| `vary-m48-F8` | §2.5 silent when all labels reach | `test_unreached_label_diagnostic.py::test_no_label_is_reported_when_every_label_reaches_an_output` |
| `vary-m48-F8` | §2.5 diagnostic, not an error | `test_unreached_label_diagnostic.py::test_the_report_is_a_diagnostic_and_the_compile_still_succeeds` |
| — | shared awkward-free fixtures | `vary_fixtures.py` |
