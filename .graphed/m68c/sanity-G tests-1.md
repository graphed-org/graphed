# m68c graphed TEST_SANITY, round 1 (suite 375ff64)

**Verdict: FAIL, 3 issues.** Checks 1 through 5 and 7 are clean for the tests themselves. Check 6
found one mechanism in the plan that no test pins (issue 1). The harness has a CI stall hazard that
cannot be fixed after the freeze (issue 2). The suite commit also breaks the lint gate through its
probe scripts (issue 3).

## Issues

1. **Nothing pins join side order (plan-graphed.md §3.3 *Builders*: "stage i reads join input i.
   Today's sorted-source-id order can swap left and right").** Every join in the suite is
   `join(ev, lu)`, and `two_sources()` registers `events` first. So a builder that keeps today's
   `sorted(partitioned.items())` stage order passes all 50 tests (mutant `sorted_sides`: 0 failed).
   It still gives the wrong answer when the left side's source was registered second:
   `join(lu, ev, how="left")` has 8 rows against `materialize`'s 9 (`p2_side_order.py`; the correct
   stand-in has 9, with rows and `.type` equal). Proposed correction: add a case where the source
   registered second is on the left, for example `join(lu, ev, how="left")` in
   `test_m68c_runs_no_service.py`, compared on `.type` and rows with `materialize`.
2. **`SFServer` calls `socket.getfqdn` on construction.** `ThreadingHTTPServer` →
   `HTTPServer.server_bind` does this; `p1_premises.py` records one call, `('127.0.0.1',)`, per
   `SFServer()`. graphed's `ci.yml` `test` job runs `macos-latest` × 4 Pythons through
   `actions/setup-python@v5`. On those runners getfqdn has been measured at 35–70 s (in executors
   m68a, fixed by `LookupFreeHTTPServer` in 0e48380; actions/setup-python#1223). This suite calls
   `serve()` 5 times per leg. The premise has not been measured on graphed's own runners. Proposed
   correction: give the harness server executors' `server_bind` override
   (`socketserver.TCPServer.server_bind(self)`, then set `server_name`/`server_port` from
   `server_address`). The harness sits under `tests/frozen`, so this has to happen before the freeze.
3. **The probe scripts committed in 375ff64 fail ruff.** These are `.graphed/m68c/probes/*.py`, and
   they are the only tracked `.py` files under `.graphed`. `ruff check .` reports 27 errors in them
   (gather_sim 20, takefix_plugin 3, reasons 2, simplified_plugin 2). `ruff format --check` flags 4
   files. The precommit gate fails on `prek` for exactly these files, and so will CI's
   `prek run --all-files ruff-check ruff-format`. Proposed correction: move them to
   `lanes/services-v2/probes/`, where other rounds keep their probes, or make them lint-clean.
   This commit adds only Markdown, which the ruff hooks do not check, so the gate failure it was
   committed under comes from these files alone.

## Checks

- **(1) Collects.** 50 items: core/m68c 6, awkward/m68c 44.
- **(2) Failure reasons on d0ad16b + tests.** All 50 fail. The first exception of each, from junit
  plus `.graphed/m68c/probes/reasons.py`, matches the evidence table line for line (sorted diff
  empty). Each one is the reason plan-graphed.md §6 states; none is an import, syntax or
  environment accident. `pytest tests/frozen/core` run as one process fails only the 6 m68c tests,
  so the m1 `awkward` guard holds. I corrected two lines of `sanity-evidence.md`'s mutant list:
  - the count-route mutant it named is not refused (`task.key` equals `task.key % 3` for keys 0 and 1);
  - a whole-file read passes `READS` and is refused by the row count instead.
- **(3) Determinism.** Two runs gave identical per-test first-exception lines. Under the stand-in,
  two runs both passed 50/50.
- **(4) Lint and types.** `ruff check` is clean and `ruff format --check` exits 0. Repo `mypy` is
  clean (583 files). With the `tests.frozen.*` override removed there are 48 errors, all from the
  API that does not exist yet (`services=`/`reduce=`, V2 passed where `Plan` is typed, `.value`,
  `.stages`), plus the one `comparison-overlap` the author noted.
- **(5) Coverage.** `--cov=graphed --cov-branch` over both dirs records hits in `shuffle.py` 88%,
  `awkward/join.py` 93%, `execute.py` 62%, `services.py` 54%, `core/plan.py` 63% and
  `core/execution.py` 55%. `run-tests.sh` collects `tests/frozen/awkward/*/` one milestone per
  process and `tests/frozen/core` as a whole, so both new dirs are in the CI coverage run.
- **(6) Discrimination.** I wrote a stand-in for §3.1–§3.4 (patch plus `M68C_MUT` mutant switches;
  `lanes/services-v2/probes/g_sanity1_standin/`, applied to a scratch copy). With no mutant it
  passes 50/50. Failures per mutant (`mutants.out`):

  | mutant | failed |
  |---|---|
  | services key always written | 1 |
  | services folded into task_id | 1 |
  | bind re-blobs the OpSpec (identity changes) | 1 |
  | endpoint not checked | 1 |
  | `value` returns a list | 2 |
  | `value` returns the last payload | 2 |
  | gather gets only its own upstream payload | 1 |
  | `LocalResources` per stage | 1 |
  | V1 counts | 1 |
  | no `require_bound` (SequentialRunner + resumable) | 5 |
  | control checked on entry only | 1 |
  | control not reset | 2 |
  | cancel returns a partial value | 1 |
  | `take` unfixed | 18 |
  | `take` `.simplified` only | 13 (per-dest file 5/6, as the author measured) |
  | broadcast honoured | 4 (left and outer) |
  | post-barrier ops dropped | 10 |
  | whole file read per task | 10 |
  | count route splits blocks | 1 |
  | untyped zero-row filler | 1 |
  | gather concatenates every dest | 10 |
  | gather calls `join_blocks` directly (grouped ignored) | 4 |
  | `services` = every declared spec | 1 |
  | chunk partials `as_outputs=True` in `join_plan` | 5 |
  | chunk partials `as_outputs=False` | 1 |
  | `target_bytes` not refused | 1 |
  | post-join source read not refused | 1 |
  | bind reaches map stages only | 1 |
  | gather gets no externals | 1 |
  | `resolve_services` applied twice | 1 |
  | no attribution hook | 1 |
  | **sorted source order for join sides** | **0 (issue 1)** |

  Main itself is the no-refusal mutant for the build-refusal tests, and they fail on it (check 2).
  One variant is left unrefused on purpose: checking control between the tasks of a stage. Plan §3.4
  allows it, and the in-stage cancel test accepts any number of map calls that is at least 1.
- **(7) Passable as planned.** The stand-in follows the plan as written, and all 50 tests pass
  under it. So no frozen m68c test contradicts the plan. The stand-in also leaves every other
  frozen suite green, except two that fail for reasons outside the plan:
  - `corpus/m05`: the scratch copy has no `docs/`.
  - `tests/extra/frontend/m39/test_shuffle_input_validation.py`: two failures are the stand-in's
    message wording. The other two are behavior the plan brings: sources now come from the map
    cone, so `shuffle_plan` over a session with two partitioned sources, and a self-join, both
    build instead of raising TypeError. That file is in extra, so the implementer updates it. Plan
    §3.3 does not decide whether a self-join should still be refused.
