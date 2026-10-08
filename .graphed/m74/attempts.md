# m74 — implementer iterations

Plan: `graphed-workdir/lanes/ckpt-resume/plan.md` (r14b) §2, §3 commits 2–3; addenda and constraints beside
it. Frozen suite: `tests/frozen/checkpoint/m74` (freeze-m74 = 5fd8ae5).

Run: `python -m pytest tests/frozen/checkpoint/m74 -p no:cacheprovider -n 8`.

## Iteration 1 (2026-10-07) — green
- `graphed.checkpoint.resume` (new): `_key_bytes` (the P14 `canonical` rule set on one
  `pickle._Pickler` subclass; payload mode dropped; `checkpoint_resolve` added), `_Keyed`, `_Resumable`,
  `_ResumableStage`, `resumable`, `check_resumable`, `StoreUnavailable`, `EnvironmentChanged`, the
  environment record check, and `open_store`/`task_store` lifted from `_PartitionReduce`.
- Stores: `record_environment` on `Store`/`FsspecStore` (own journal `journal.environment.log`);
  `_replay` returns `Completed`, a dict with the set-aside `environments`.
- `run_resumable`: streaming fold, environment check; `run_shuffle_resumable`: environment check after
  `require_bound`; both refuse a store without `record_environment` before any store call.
  `_partition_tag` now lives in `resume.py`.
- Declarations: `checkpointable`/`checkpoint_ignore`/`checkpoint_resolve` on `_PartitionReduce`,
  `_MapWrite`, `_Gather`, `_Fold`, `_WritePart`, `_VariedWritePart`, `CompiledGraph`; `_Collated` as a
  property. `_resolve_behavior` moved above `_WritePart` (its class body names it).
- Extra: `tests/extra/checkpoint/m74/test_m74_extra.py` (9) for paths the frozen suite reaches only in
  subprocesses or not at all; 8 single-site mutations of `resume.py` each fail one of them.
- Addenda: two lines (§2.4 store cache, §2.6 stage records).
- Gates: m74 frozen 75/75; `COV=1 run-tests.sh` rc 0; ruff, mypy --strict (+ win32) clean; sphinx -W ok
  (docs example executed); frozen diff vs freeze-m74 empty.

## Iteration 2 (2026-10-07) — precommit integrity, green
- `integrity-scan` flagged four `# type: ignore` on the pure pickler's hooks (typeshed declares none of
  them); the hooks are now called through one `Any`-typed alias of `pickle`, no ignores left.
- Extra: `_resolve_behavior`'s import-ref branch (moved code) covered in process.
- Gates: m74 frozen 75/75; `COV=1 run-tests.sh` rc 0; touched files 97–100%; diff-cover vs eea2f8d 100%
  (376 lines); precommit static checks ok, sphinx -W ok. The precommit `pytest` fallback collects the
  whole tree in one process and fails collection on helper-basename collisions (`analyses`,
  `shuffle_backends`) unrelated to m74; `run-tests.sh` is this repo's runner.

## Iteration 3 (2026-10-07) — review fixes (impl1-findings: m74-D1, N1, N2, T-1, T-2, D-1)
- m74-D1: every read of cloudpickle's private API is in `_cloudpickle()` (cached, called from
  `_KeyPickler.__init__`); no class attribute or module global binds one. A missing name makes
  `check_resumable`/`resumable` raise `TypeError` naming cloudpickle; `import graphed.checkpoint`,
  the stores and `run_resumable` keep working. Extra test: one subprocess leg per deleted private name
  plus a control; both deletion legs fail on da9c20c.
- N1: the key pass pickles into a sha256 sink (`_key_digest`), so key memory stays flat. The memory
  regression was this iteration's own eager label in the sink version; da9c20c already built the
  process repr only in `except`, so the lazy `_of` label fixes nothing there. Element sort keys keep bytes.
- N2: each `ak.behavior` entry is keyed alone; an unkeyable one is refused as `ak.behavior[<key>]`.
- T-1: the vacuous tracker test is replaced by by-value TypeVars of different names; it kills
  "blank every str", "no blanking" and "`_TRACKED = (type,)`". T-2: two tests renamed. D-1: the two
  m74 headings no longer nest a literal in bold.

## Iteration 4 (2026-10-07) — delta-review follow-ups (m74-r2-N1, N2, N3)
- r2-N1: `_cloudpickle()` reads the private names listed in `_CLOUDPICKLE_PRIVATE`; the subprocess
  test is parametrized over that list, so an import-time read of any of them fails its leg (the
  reviewer's `_module_reduce`, `dynamic_subimport` and `Pickler._dispatch_table` mutants now die).
- r2-N2: `_task_ids` calls `_cloudpickle()` before keying anything, so a missing name refuses with the
  cloudpickle message alone; the in-process test registers a keyable `ak.behavior` entry and asserts
  the exact message.
- r2-N3: Iteration 3's N1 line corrected.
