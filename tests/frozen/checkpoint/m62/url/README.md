# m62 unit B frozen suite — `FsspecStore`, the checkpoint store at a URL

Frozen at `freeze-m62` (alias `freeze-m62b`); read-only afterwards. Spec: lane plan `plan-B.md` §B.3–B.4.

New API: `graphed.checkpoint.FsspecStore(url, node=None, **storage_options)`, a `CheckpointStore` with
`url`, `node`, `fs`, `root`, `objects`, `journal_path`, `dead_letter_path`; with fsspec not importable
it raises an `ImportError` naming `graphed[checkpoint]`.

Fixtures (`conftest.py`, unit C's contract): `store_url` (params `memory`, `file`; fresh root per test)
and `shared_url` (fresh `file://` root a child can reach). No other fixture, no module `pytestmark`;
shared code is plain functions in `m62_url_helpers.py`. Unit C re-collects every test that takes one of
the two fixtures, except TB4.

| Test | Pins | Witness | Wrong implementation it rejects |
|---|---|---|---|
| `test_url_store_seam.py` TB1 `test_fsspec_store_satisfies_the_protocol` | B.3.1 | `isinstance`; attribute paths; `root == url_to_fs(url)[1]`; storage options reach `url_to_fs` (`skip_instance_cache=True` → a new `fs`) | options dropped; wrong layout |
| TB2 `test_url_get_refuses_bytes_that_do_not_hash_to_their_name` | B.3.2.2 | tampered object → `None`; re-`put` rewrites the object | unverified `get`; presence-guarded `put` |
| TB3 `test_url_resume_recomputes_a_corrupted_partial_and_heals_it` | B.3.2.9 | `executed == 1`, reference value, healed blob hash, second resume `executed == 0` | no verify; verify without heal |
| TB4 `test_url_concurrent_identical_puts_never_fail` | B.3.2.6 (puts) | 20 trials × 8 threads × 1 MiB, one object named by the digest | exclusive-create put (`FileExistsError`, file leg) |
| TB5 `test_url_put_keeps_the_default_file_mode` | B.3.2.14 | blob mode == sibling mode under `umask(0o022)` | `mkstemp` temp (0600) |
| TB6 `test_root_with_glob_metacharacters_keeps_its_records` | B.3.2.13 | root `<store_url>/run[x]`, default + `node="B"` writers, union replay and dead letters | unescaped glob; `fs.cat(list)` |
| TB25 `test_local_root_with_glob_metacharacters_keeps_its_records` | B.3.2.13 for `Store` | same helper on `Store(tmp/"run[x]")` | string-glob `Store.completed` |
| `test_fsspec_store.py` TB7 `test_put_is_content_addressed_and_idempotent` | B.3.2.1 | `fs.find(objects)` holds one object named by the digest | non-content names; duplicate objects |
| TB8 `test_get_missing_blob_is_none` | B.3.2.1 | `None` on empty and non-empty stores | — |
| TB9 `test_fresh_store_is_empty` | B.3.2.7 | `{}` and `[]` | `ls` presence (s3 re-run) |
| TB10 `test_journal_replays_stage_deps_and_last_record_wins` | B.3.2.3 | 12 records for one task through one instance: the last `JournalEntry` (stage, deps) wins | first wins; unpadded names |
| TB11 `test_record_without_its_blob_is_not_honored` | B.3.2.3 | blobless record dropped, blobbed record kept | presence not checked |
| TB12 `test_unparseable_record_is_skipped` | B.3.2.3 | the M8 torn line as its own object | parse error raised |
| TB13 `test_records_are_one_line_objects_with_the_local_bytes` | B.3.2.5 | every record object is one line; sorted bytes equal a local `Store` fed the same calls (default and `node="A"`) | a serializer whose bytes differ |
| TB14 `test_dead_letters_keep_insertion_order` | B.3.2.4 | 13 records in order, same and fresh instance | unpadded sequence names |
| TB15 `test_node_writers_replay_as_a_union` | B.3.2.5 | one object under each of `journal.A.log/`, `journal.B.log/`, none under `journal.log/`; union replay | single-prefix replay |
| TB16 `test_concurrent_record_dead_loses_nothing` | B.3.2.6 (records) | 8 threads × 10 records incl. 40 identical: counted by value, 80 objects | content- or task-keyed names |
| TB17 `test_unreadable_record_raises_the_backend_error` | B.3.2.16 | patched `fs.cat_ranges` returns `boom` last of two paths; both readers raise that object | `on_error="raise"` without the scan; first-element check; widened catch |
| `test_remote_resume.py` TB18 `test_kill_then_resume_equals_uninterrupted` | B.3.2.8 | `skipped == 4`, `executed == 2`, reference value, fresh instance holds all task ids | pid-keyed writer id |
| TB19 `test_shuffle_kill_then_resume_equals_uninterrupted` | B.3.2.8 (m39 `shuffle_analyses`) | `skipped == 3`, value == local reference, fresh key set == local key set | pid-keyed writer id |
| `test_remote_crossprocess.py` TB20 `test_other_process_resumes_from_the_url_alone` | B.3.2.10 | child gets the plan file's path and the URL; `(2, 4)` and reference value; the pre-listed parent sees all ids and verified blobs | listing cache (s3 re-run) |
| TB21 `test_other_process_finds_nothing_left_to_do` | B.3.2.10 | child `(0, 6)` | — |
| `test_store_determinism.py` TB22 `test_plan_bytes_and_task_ids_do_not_depend_on_the_store` | B.3.2.11 | plan bytes stable; task-id sets; sorted record bytes and value bytes equal across `Store` and `FsspecStore` | store-dependent plan or records |
| TB23 `test_import_does_not_load_fsspec` | B.3.2.12 | child prints `False True` | eager fsspec import |
| TB24 `test_missing_fsspec_names_the_extra` | B.3.2.15 | `fsspec` and every loaded `fsspec.*` set to `None`; message matches `graphed\[checkpoint\]` | bare re-raise |

## TEST_SANITY (lane venv, CPython 3.13.3, macOS, fsspec 2026.9.0)

- Stub: an `FsspecStore` whose constructor raises, injected by a pytest plugin. No test is skipped. Every
  `FsspecStore` test fails: 40 in-process legs on the stub's constructor, and the TB20/TB21/TB23 children
  on `FsspecStore` missing from `graphed.checkpoint`. TB25 passes (it pins `Store`).
- TB25's mutant, run once: `glob.glob(str(self.root) + "/journal*.log")` in `Store.completed` fails it
  (`set() == {'task-A', 'task-B'}`).
- Positive control and mutants: a scratch contract-following `FsspecStore` (never committed) passes the
  whole suite. Each named mutant of it fails the tests the table names: unverified get (TB2, TB3),
  presence-guarded put (TB2, TB3), unpadded names (TB10, TB14, …), pid-keyed writer (TB18, TB19, …),
  content-keyed names (TB16, …), exclusive-create put (TB4[file]), `fs.cat(list)` (TB6), bare re-raise
  (TB24), and the three TB17 read mutants (TB17). The TB5 `mkstemp` mutant was not run here; TA6 shows
  the same assertion rejecting it.
- Two runs of the whole checkpoint process give identical per-test outcomes. ruff and mypy clean.
