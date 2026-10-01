# m68c graphed TEST_SANITY, round 2 (suite 443f9ac)

**Verdict: PASS, no issues.** All three round-1 issues are closed. The branch changes only
`tests/frozen/{core,awkward}/m68c/`, `.graphed/m68c/` and the pyproject mypy override, so running
the worktree is running d0ad16b + tests.

## Round-1 issues
1. **Join side order:** closed. `test_join_plan_keeps_the_later_registered_source_on_the_left` is
   refused by the `sorted_sides` mutant (round 1: 0 failed; now 1 failed). Under the stand-in the
   correct builder gives 9 rows and `.type` equal to `materialize`; the mutant gives 8
   (`p2_side_order.py`).
2. **getfqdn:** closed. `SFServer()` now makes 0 `socket.getfqdn` calls. A plain
   `ThreadingHTTPServer`, used as the control, makes 1. `serve()` still serves, and the stand-in
   passes 51/51 through it.
3. **Probe lint:** closed. No `.py` is tracked under `.graphed/`. The precommit gate
   (`--fast --no-coverage`) exits 0: toml, workflows, integrity-scan, prek, and mypy strict all ok.
   It needed no `--allow-refreeze`.

## Checks
- **(1) Collects:** 51 items (core/m68c 6, awkward/m68c 45).
- **(2) Failure reasons on d0ad16b + tests:** 51/51 fail. The first exception of each, from junit
  and `g_author/reasons.py`, matches `sanity-evidence.md`'s table line for line (sorted diff empty),
  so that file needs no correction. The new test fails with AttributeError `services` raised in
  `SequentialRunner.run` → `require_bound`, which is §6 *Runs with no service*' stated reason.
  Running `pytest tests/frozen/core` as one process gives 6 failed and 194 passed: only m68c fails,
  and the m1 awkward guard holds.
- **(3) Determinism:** two runs on the base give identical per-test first-exception lines. Two runs
  under the stand-in both pass 51/51.
- **(4) Lint/types:** `ruff check` and `ruff format --check` are clean on both dirs and `.graphed`.
  `mypy` passes on the repo config (583 files) and on the two dirs (14 files).
- **(5) Coverage:** `--cov=graphed --cov-branch` over both dirs records hits in `shuffle.py` 88%,
  `execute.py` 62%, `services.py` 54%, `core/plan.py` 63% and `core/execution.py` 55%.
  `run-tests.sh` collects `tests/frozen/core` and `tests/frozen/awkward`.
- **(6) Discrimination:** the round-1 stand-in (`probes/g_sanity1_standin/standin.patch`) was
  applied to a scratch copy of 443f9ac and all 32 mutants were rerun against it
  (`probes/g_sanity2/mutants.out`). Every mutant is refused by at least one test. `take_nofix`,
  `whole_file` and `gather_all` each fail one more test than in round 1, and that test is the new
  one. The only variant left unrefused is the one round 1 named: checking control between the
  tasks of a stage, which §3.4 allows.
- **(7) Passable as planned:** the §3 stand-in passes all 51 tests, so no test contradicts the plan.
