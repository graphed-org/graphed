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

## C2 — the frontend per-label verbs + §2.5 shift-after-weight (iteration 1)

Targeted: `frontend/m49/test_impact_sets.py` (12), `test_varied_read_columns_projection.py` (the 3
stats/mapping arms), `test_record_correspondence.py::test_two_labels_sharing_a_node_collapse_onto_one_key`
(C1's anchor, which reads §3.4's verb), `awkward/m49/test_shift_after_weight.py` (3).

Design as bound by plan §3.4/§5.3/§2.5: one new module `python/graphed/by_label.py` holds both
verbs and the operand form they share, because §3.4 binds ONE rejection contract for both — a
sequence must be all `Varied` with no nested member, a mapping must map a label to a sequence of
`Array`, and each refusal names the offending element's type (and the key, on the mapping form).
The resolved operand is `{label: that label's Arrays}` over the §2.4 union (`union_labels` +
`member_of`, never the strict `graphed.universe`); `impact_by_label` takes the reachability
difference against the nominal entry's cone, `read_columns_by_label` applies the existing
`read_columns` per label so `None` keeps meaning "read every column". `graphed.member_of` is
re-exported from `varied` (no annotation mentions `Array`, so the §2.3d gate does not discover it);
both verbs enter `VERB_DISPOSITIONS` as *expanding*, which §2.3d's m49 clause pre-authorises.

§2.5's shift-after-weight is record-time and by value: `_vary_weight` appends
`(family, that factor's own member node ids)` to `Session._weight_factors`, and `_vary_shift` walks
each recorded factor's cone against the PRE-shift collection's node ids, adding
`(family, collection)` to `Session._shift_after_weight`. `compile_ir` reports it sorted as the
second additive `CompiledGraph` field, after `correspondence`.

Gates: 13 frozen failures, all pre-existing and all C3's (`frontend/m49/test_boundary_refusal`,
`awkward/m49` join-refusal + broadcast-blame, `debug/m49`); zero new; exactly the 19 targeted turned
green. mypy strict clean; ruff check + format clean; combined branch coverage 94% with
`by_label.py` at 100% and every new `context.py`/`execute.py` line covered from the FROZEN suite;
determinism suites and the `core/m49` §3.3 benchmark green; `precommit . --fast` = ok. No Rust
touched. +166 LOC.
