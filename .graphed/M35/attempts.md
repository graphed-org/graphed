# M35 attempts — graphed (Session.walk is iterative)

## Iteration 0 — 2026-06-13 (freeze-M35-0)

- Review finding P1-4: Session.walk (shared by materialize + projection) evaluated recursively
  (ev calls ev), so a deeply-chained recorded graph -- a long selection/systematics chain --
  would RecursionError past Python's ~1000-frame limit. The hot executor runs the SHALLOW
  reduced graph so production was safe, but the reference/eval path was latently broken on deep
  graphs.
- FIX: explicit-stack post-order traversal; inputs pushed reversed so the first input resolves
  first (left-to-right compute order, matching the prior depth-first recursion); the cache
  deduplicates shared sub-DAGs. Behavior is identical for all existing graphs (every materialize
  test unchanged); only the recursion ceiling is removed.
- frozen m35 (2): a chain of depth 3x the recursion limit materializes to the correct value
  with no RecursionError (non-vacuous: the recursive impl raised here); a diamond/shared
  subgraph still evaluates once and correctly. Gates green via the precommit script.

## Iteration 1 — 2026-06-13 (freeze-M35-1)

- CI caught what local could not: m35 imported graphed_numpy, but the graphed repo depends only
  on graphed-core (graphed-numpy is downstream, NOT installed in graphed's CI) -> ModuleNotFound
  on every cell. Local passed because graphed-numpy is editable-installed in the dev venv.
- Sanctioned correction (own just-frozen, CI-broken milestone): rewrote m35 on the
  backend-independent ListBackend (tests/frozen/m2/backends.py, already on the pythonpath; pure
  Python lists, no numpy/graphed_numpy). Deep chain = N elementwise list-adds; diamond = shared
  list-add reached by two paths. Same property pinned (iterative walk past 3x recursion limit).
  Re-frozen freeze-M35-1. Lesson: a frozen test must use only the repo's OWN dependencies; the
  precommit gate runs in the dev venv (all siblings present) so it cannot catch a missing-dep
  import that CI will.
