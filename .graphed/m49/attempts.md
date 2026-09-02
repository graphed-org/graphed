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

## C3 — the error path (iteration 1)

Targeted: `frontend/m49/test_boundary_refusal.py` (4), `awkward/m49/test_join_refusal.py` (2) +
`test_broadcast_blame.py` (2), `debug/m49/test_variation_attribution.py` (4) +
`test_variation_process_boundary.py` (1) — the 13 reds C2 left.

Design as bound by plan §8.1/§8.2(ii)(iii)/§5.4/§6.1d: `StageError` gains `variation: str = ""`
(summary clause suppressed when empty, added to the hand-written `__hash__` tuple; the existing
`__dict__` `__reduce__` carries it across a process boundary unchanged). `evaluate_ir` gains an
optional `on_failure` hook; both dispatch points route through one `_dispatch` helper that annotates
the failure with `(reduced_node_id, member_index | None)`. `_PartitionReduce.__call__` supplies the
hook: an entry for the failing key in `variation_labels` becomes a `StageError` with that entry's
labels (sorted, comma-joined; the empty tuple falls out as `""`) and its frame, no entry re-raises
untouched. §5.4's message is widened as a class through one `boundary_refusal` helper shared by all
three spellings; `refuse_container`'s m48-anchored wording is untouched. §6.1d's blame lives in
`_ops.apply`'s `ak.broadcast_arrays` arm — the one site `op_form` and `eval_stage` share, so the
record-time and execution-time legs are translated together.

Gates: full `COV=1 ./scripts/run-tests.sh` = 0 failures over 55 processes (the `graphed` frozen red
set is now EMPTY); `comm -13` against the pre-C3 FAILED set = 0 new, `comm -23` = the 13 targeted.
Combined branch coverage 94%; every new line in the five touched modules covered, the residual
misses in those files pre-existing. Determinism suites and the `core/m49` §3.3 benchmark green;
the spawned-process picklability anchor green. mypy strict clean; ruff check + format clean;
`precommit . --fast` = ok. No Rust touched. +143 src LOC, +98 in `tests/extra/{frontend,awkward}/m49`
(the top-level dispatch point and the unflatten hint's discrimination, neither witnessed by the
frozen suite; each verified by a restored mutant).

## Fix cycle 1 — review wf_fa3d406b (A-1/A-3/A-4/A-8/A-11/A-12/A-13)

**A-3 (MED, shipped)** — `CompiledGraph.shift_after_weight` is now per-PROGRAM. The design lens's
proposed record-site edit was NOT made (the adjudicator's correction: `_report_shift_after_weight`
is plan-faithful, §2.5 binds the check to the session-recorded families and their cone walk); the
defect was the shipping site, so `Session._shift_after_weight` became `{(family, collection): the
offending factor's member node ids}` and `compile_ir` keeps a pair only when those ids intersect
this artifact's `node_map`. Both witnesses close, the three frozen anchors stay green.

**A-4 (MED, shipped)** — both `by_label` verbs spell `Sequence[Varied] | Mapping[str,
Sequence[Array]]` inline; under `from __future__ import annotations` the `Outputs` alias
stringified to its own name and hid the `Array` mention §2.3d's filter reads. The alias comment
claimed the opposite and is corrected. Closing witness is the ALREADY-FROZEN m48 anchor:
`_discovered()` went from `False/False` to `True/True` for the two verbs, with
`("graphed","read_columns")` `True` throughout as the live control and `("graphed","cone")` `False`
as the negative one; `tests/frozen/awkward/m48/test_module_verb_dispositions.py` 13 passed.

**A-8 / A-11 / A-12 / A-13 (shipped)** — `tests/extra/frontend/m49/test_unoptimized_correspondence.py`
covers the `optimize=False` correspondence arm (identity map + still evaluates) against the
optimized artifact of the same program as its control; `impact_by_label` answers `nominal` from the
`central` walk it already did instead of re-walking; `CompiledGraph.__hash__ = None` states the
unhashability a dict field already caused (`hash()` now says `CompiledGraph`, not `dict`);
`from_reduced`'s re-key comment no longer claims a merge no input is known to produce — the remap
is the same `map` the inputs and outputs ride, so there is nothing separate to delete or witness.

**A-1 (HIGH) — NOT shipped, TEST DISPUTE filed.** Routing `evaluate_ir`'s `external` arm through
`_dispatch` reds four frozen anchors in graphed-histogram's `freeze-m49` suite
(`tests/frozen/m49/test_blame_parity.py`): §6.1d pins the plan path to re-raise an External
evaluator's own `GraphedError` verbatim, which attribution replaces with a `StageError` (a bare
`Exception`, so `pytest.raises(GraphedError)` does not even catch it). Measured both ways on one
fixture: repair -> EXIT=1/4 failed; freeze behaviour -> EXIT=0/5 passed, while `graphed`'s own
suite and `graphed-executors tests/frozen/m49` are green under both. Not routed around, not
weakened: `graphed-histogram/.graphed/m49/disputes/test_blame_parity.md` carries the claim, the
measurement and the ready-to-freeze witness; the `external` arm keeps a comment pointing at it.

Gates: `COV=1 ./scripts/run-tests.sh` 0 FAILED, combined branch coverage 94%; `core/m49` benchmark
green; ruff check + format clean; `mypy --strict` clean; `cargo clippy -D warnings` clean and
`cargo test --release` 30 passed (comment-only Rust change); `precommit . --fast` ok with
`workflows-valid` now LIVE (`ok`, not the `pyyaml not installed` skip C0-C3 recorded).
