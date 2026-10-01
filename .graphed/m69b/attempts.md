# m69b implementer attempts — `aggregate_plan(opt_level=)` (plan-services §5.0)

## Target
Frozen `tests/frozen/frontend/m69b` (12 tests, tag `freeze-m69b` c1174a8). Baseline at the tag
(src = d0ad16b), `./scripts/run-tests.sh`: 2647 passed, 33 skipped, 12 failed — exactly the 12
m69b tests.

## Iteration 1 — the cone, the level, one-compile replay
- Rust `GraphStore::cone(outputs)`: refuses an id past the arena (`BadNodeId`, shared with
  `reduce_with_outputs` through `arena_holding`), runs `optimizer::dead_code_elimination`, rebuilds
  through `from_reduced` with `node_map` = record id → `(cone id, None)`. Binding
  `GraphStore.cone(*, outputs)`; cargo test pins the bytes against a hand-built store, the
  node_map, no store state written, and the refusal.
- `execute._compile_cone` shares `compile_ir`'s tail (`_compiled`: node_map, frames by key,
  labels). `aggregate._compile_at(level, ...)` is the one switch; `_PartitionReduce.opt_level`
  carries the level into `StageError` and states the rule. `aggregate_plan` refuses levels other
  than 0/1 before anything else; `_slot_of` lifted out of `aggregate_plan` for replay.
- `replay()` compiles once at `process.opt_level` and hands that `CompiledGraph` to `Replay`, which
  reduces one value per IR output (first output landing on each slot). Direct `Replay(...)` still
  works: it compiles at the process's level itself.
- Frozen 12/12 on the first run. Extra `tests/extra/frontend/m69b` (direct Replay at 0 and 1);
  mutant "fallback compiles at 1" fails `[0]`, restored passes. Rust mutant "no id check" panics
  the cargo test.
- Gates: run-tests 2661 passed, 33 skipped, 0 failed (COV=1); touched files aggregate 99.62%,
  replaying 100%, execute 97.53%, store.rs 96.51% (llvm-cov, LLVM 22); diff-cover 100% Python
  (32 lines) and Rust (46 lines); the 4 files under 90% locally are the torch/tf/jax/xgboost
  plugins (frameworks not installed here; CI installs them). prek ruff/format/mypy/cargo-fmt/
  clippy, cargo test 51, loom, sphinx -W, precommit --fast --no-coverage: all ok.

## Iteration 2 — review r1 Lows (impl-m69b-A-r1.md)
- L1: `opt_level` `True`/`False`/`0.0`/`1.0` passed the equality check and were stored as given
  (probed). The check now also requires `type(opt_level) is int`; extra
  `test_m69b_level_type.py` fails 4/4 on the old check, passes on the new.
- L2: `_PartitionReduce.opt_level`'s comment cut to the field's meaning.
- Gates: frozen m69b 12/12; frontend (per-milestone) + extra frontend + debug subtrees green;
  touched-file coverage over those subtrees aggregate 98% / replaying 100% / execute 98%;
  diff-cover vs origin/main 100% (32 lines); prek ruff/format/mypy/cargo-fmt/clippy and
  precommit --fast --no-coverage ok.
