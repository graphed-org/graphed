# m49 implementation attempts log
Newest at the end. Pipeline: root CLAUDE.md Part B; frozen suites = tags m49-freeze / freeze-m49.

## C1 — record→reduced correspondence (iteration 1)

Targeted: `frontend/m49/test_record_correspondence.py` + the correspondence-shaped half of
`debug/m49` (`artifact_frames` layout) + no regression on the already-green `checkpoint/m49`.

Design as bound by plan §8.2(i): each of the four reduction passes returns the re-indexing it
already computes (`dead_code_elimination` its `remap`, `RewriteEngine::canonicalize` a total
`node_map`, `cse` its `remap`, `stage_fusion` each node's `(reduced id, member index)`), and
`reduce_with_mode` composes them. `IncrementalReducer::finalize` composes its own
original→canonical map in front. The composed map rides `Reduced` into `GraphStore::from_reduced`,
which re-keys it through the re-interning and parks it on the reduced store
(`GraphStore::node_map()` / PyO3 `PyGraphStore.node_map()`) — no `reduce*` tuple arity changed, so
no frozen core test that destructures `(store, report)` was disturbed. `compile_ir` turns it into
`CompiledGraph.correspondence` (plain, non-slots dataclass) and re-keys `Session._provenance` onto
the same keys with §8.2(ii)'s lowest-record-id tie-break.

Gates: 32 frozen failures, all pre-existing (baseline 37); zero new. Fixed exactly the 5 listed in
the worklog. cargo test 30 passed; clippy/fmt clean; mypy strict clean; ruff clean;
combined branch coverage 94%; `precommit . --fast` = ok. Determinism + §3.3 benchmark
(`core/m49`) green.
