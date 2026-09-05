# m52 graphed implementer — iteration log

Freeze: `m52-freeze`. Scope C0–C7 (graphed: C0,C1,C2,C3,C4,C6-corpus-fixture consumer,C7).
`git diff m52-freeze -- tests/frozen/` MUST stay empty for the life of the milestone.

## Iteration 0 — baseline (origin/main, no m52 implementation)
- frontend/m52: 43 failed / 2 passed (the 2 green are the §8-b partial-coverage + m50 kind-guard
  backward-compat positive controls).
- awkward/m52:  9 failed / 1 passed (the green is §8-j's boundary half, `test_a_default_point_
  registration_still_reduces_a_different_tag_or_family_to_nominal` — C3 must NOT change it).
- corpus/m52:   3 failed / 60 passed (corpus/m05's 60 untouched and green).
Baseline failure reasons match the (adjudicated) decomposition §5 table: `points=`-fixture anchors
raise `GraphedError: ...got dict` at origin/main (feature absent); C1-row anchors raise
`ModuleNotFoundError: graphed._points`; introspection rows `AttributeError: no 'points'`; the two
default-point awkward anchors 3.5 (fail on inverted relation) / 3.6 (pass).

## Iteration 1 — C1 (`graphed/_points.py`)
New `python/graphed/_points.py`: `Point` (name-sorted pair tuple, explicit-map zero-drop),
`default`, `restrict`, `render`, `coordinate`. Numeric coordinates are stored already reduced to
their VALUE's e-form through `_tags._normalize`/`_render`, so equality across spellings and
render-by-value fall out of one representation. Imports `._tags` + `.errors` only.
- frontend/m52 `test_point_value.py`: 0 failed / 13 passed.

## Iteration 2 — C2 + C3 + C4
- `session.py`: `Session._points: dict[str, Point]` (plain attribute, `Session` has no `__slots__`).
- `varied.py`: `session_of` / `point_registry` / `registered_points` (registry reached through the
  nominal member, `session` being RESERVED); `Varied._member_for` keeps its fast path and its
  FALLBACK branch restricts -> looks up -> falls back to nominal.
- `vary.py`: `points=` keyword-only on all three overloads; the transactional mint (snapshot of
  `Session._points`, restored on any exception out of the overload); `_mint_points` +
  `_reachable`/`_check_reachable` (§4.11-4) + `_check_unique` (§4.11-1/2) inside `gather_members`;
  `central_universe` DELETED, both call sites routed through `member_of`.
- `context.py`: `points` threaded through `vary_context`/`_vary_weight`/`_vary_shift`; `_carriers`
  is the three `_context_labels` reads; `central_universe` dropped from the import.
- `accessors.py`: `points()` (label-sorted, nuisance-sorted, by VALUE, refusing a result mapping);
  `variations()` reports `"both"` per (name, tag) present in the ambient tag map AND a collection's.
- `__init__.py`: `points` exported.
`_two_level` and `_follow`'s mask branch needed NO edit — both already narrow through
`member_of`/`narrow`, so they became point-aware with `_member_for`. Asserted, not assumed:
`test_two_level_reaches_the_true_inner_cross_member` and `test_reindex_to_follows_a_label_by_its_
points_own_mask` are green.
- frontend/m52: 0 failed / 45 passed. awkward/m52: 0 failed / 10 passed.
- every pre-existing frozen suite green under `scripts/run-tests.sh`.

### Two non-frozen repairs this iteration needed
1. `conftest.py` (NEW, repo root) exports `pyproject.toml`'s `pythonpath` roots into `PYTHONPATH`.
   `awkward/m52::test_the_joint_program_is_byte_deterministic_across_two_runs` re-runs the program
   in a FRESH interpreter that inherits `os.environ` alone, so its child could not import the
   vendored `graphed_corpus` the parent reads off the pytest `pythonpath` (measured:
   `ModuleNotFoundError: No module named 'graphed_corpus'` from the child, and the anchor passes
   with `PYTHONPATH=tests/_corpus` set). The two clean-machine anchors that must not see this tree
   scrub `PYTHONPATH` themselves, so their isolation is unaffected.
2. `tests/extra/awkward/m48/test_row_space_and_memoisation.py` (extra, not frozen): five
   cross-name prefix collisions now refuse at MINT time under §4.11-1 rather than at the m48
   carrier-keyed check, with the §4.11-1 message. The refusal is preserved and Session-wide; only
   the `pytest.raises(match=)` text moved, to `"one label names one universe"`.

The one remaining red in the whole tree is `frozen/corpus/m52::test_the_joint_universe_is_not_the_
factorized_product_and_equals_the_direct_reference`, blocked on C6's corpus half
(`AttributeError: module 'graphed_corpus.analyses.systematics' has no attribute
'ttbar_joint_reference'`) — not a C1-C4 target.
