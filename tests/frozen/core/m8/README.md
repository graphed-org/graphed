# M8 frozen suite — graphed-core (plan serialization)

graphed-core owns the **M8-plan** half of M8: the canonical, versioned, byte-identical durable IR
form and the content-addressed `DurablePlan`. The Store / checkpoint / resume / error-harvesting
half lives in `graphed-checkpoint`.

| Test (file) | Plan clause it pins |
|---|---|
| `test_ir_serialization.py::test_roundtrip_preserves_structure` | canonical durable form is the serializable IR (A.3.1) |
| `…::test_reserialize_is_byte_identical`, `…::test_identical_graphs_serialize_identically` | "Identical plan → byte-identical serialization" (M8 gate) |
| `…::test_serialization_is_versioned` | "Versioned deterministic Plan serialization" (M8 target) |
| `…::test_reduced_graph_with_stages_roundtrips` | the durable plan carries the *reduced* IR (Stage nodes) |
| `…::test_external_payload_descriptor_survives_roundtrip` | `External` payload descriptors are durable reproducibility metadata (A.3.1) |
| `…::test_bad_magic_is_rejected`, `…::test_truncation_is_rejected` | a corrupt/foreign blob never silently decodes |
| `test_durable_plan.py::test_serialization_is_byte_identical_for_identical_plans` | M8 determinism gate at the plan level |
| `…::test_roundtrip_recovers_every_field` | the Plan carries read_columns/partitions/reduction/stopping/locality/resource (M8 target) |
| `…::test_task_id_is_deterministic_and_per_partition`, `…::test_task_id_changes_with_the_computation`, `…::test_task_id_changes_with_the_process_callable` | "Is `task_id` actually content-addressed (cache-poisoning-safe)?" (M8 review focus) |
| `…::test_importable_callable_is_referenced_not_pickled`, `…::test_only_opaque_callables_are_embedded_by_value`, `…::test_plan_opaque_flag_reflects_any_opaque_callable`, `…::test_opaque_process_changes_task_id_and_roundtrips` | canonical form is IR; cloudpickle only for opaque, flagged `opaque=True` (A.3.1) |
| `…::test_plan_resolves_and_runs_with_no_user_source_files` | "A serialized plan deserializes and runs on a machine with NO source files present" (M8 contract) |
| `test_deployment.py::test_partition_dataset_*`, `…::test_partition_datasets_concatenates` | `Dataset` → partitions builders (chunking, exact cover, edge cases) |
| `…::test_with_partitions_shares_the_compiled_computation`, `…::test_compile_once_is_reused_across_many_datasets` | "compile once, run on N datasets": the optimized interned IR is reused unchanged when re-targeting |
| `…::test_task_ids_are_namespaced_per_dataset` | per-dataset `task_id`s are disjoint so one checkpoint store namespaces datasets safely |
| `…::test_for_dataset_and_for_datasets_build_a_deployment`, `…::test_retargeting_only_changes_partitions_in_the_serialized_form` | `for_dataset`/`for_datasets` build a deployment; same computation, different inputs |

Frozen = read-only after the freeze tag (see `.graphed/M8/`).
