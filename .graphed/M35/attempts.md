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
