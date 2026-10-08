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
