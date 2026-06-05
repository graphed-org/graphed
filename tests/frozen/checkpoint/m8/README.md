# M8 frozen suite — graphed-checkpoint (checkpoint / resume / error harvesting)

graphed-checkpoint owns the **durable** half of M8 (the Store, resume, and error harvesting). The
serializable `DurablePlan` + content-addressed `task_id` half is tested in `graphed-core`'s M8 suite.

| Test (file) | Plan M8 clause it pins |
|---|---|
| `test_store.py::test_put_is_content_addressed_and_idempotent` | content-addressed `Store`; the blob name is its hash (cache-poisoning-safe) |
| `…::test_journal_entry_without_its_blob_is_not_honored`, `…::test_torn_trailing_journal_line_is_ignored` | resume is correct under **partial/interrupted writes** (review focus) |
| `…::test_dead_letter_set_is_append_only` | append-only dead-letter set |
| `test_resume.py::test_uninterrupted_run_matches_single_pass` | baseline: chunked == single pass |
| `…::test_kill_then_resume_equals_uninterrupted_and_does_less_work` | "Kill halfway; re-run; equals an uninterrupted run AND does measurably less work (skipped logged)" |
| `…::test_resume_after_completion_redoes_nothing`, `…::test_no_double_count_when_kill_lands_between_partitions` | **no double-count, no lost partition** at any crash boundary |
| `…::test_result_is_invariant_to_partition_count` | the histogram is invariant to how events are chunked |
| `test_errors.py::test_injected_failure_lands_in_dead_letter_with_reproducible_descriptor` | "injected per-partition failure lands in the dead-letter set with a reproducible descriptor" (+ M6 provenance) |
| `…::test_retry_smaller_chunk_succeeds_where_whole_chunk_oomed` | "`retry_smaller_chunk` succeeds where the original OOMed (simulated)" |
| `…::test_retry_n_recovers_a_transient_failure`, `…::test_retry_elsewhere_reruns_on_a_fresh_worker_context`, `…::test_exhausted_retries_still_dead_letter`, `…::test_smaller_chunk_gives_up_on_a_non_size_failure` | policies `retry_n` / `retry_smaller_chunk` / `retry_elsewhere` / `quarantine` |
| `…::test_error_budget_stops_the_run` | error budget as a stopping condition |
| `test_no_source.py::test_serialized_plan_runs_with_no_user_source`, `…::test_resumed_subprocess_plan_skips_completed_work` | "A serialized plan deserializes and runs on a machine with NO source files present" |
| `test_codec.py::*` | deterministic per-task partial storage (pickle + numpy `.npy`) |
| `test_realistic.py::*` | kill/resume on a **real** graphed analysis (ADL-q1 MET pt), bit-for-bit |

Frozen = read-only after the freeze tag (see `.graphed/M8/`).
