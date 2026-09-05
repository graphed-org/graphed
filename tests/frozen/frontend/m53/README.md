# frontend/m53 — dependency-fanout policy knobs (traceability)

Awkward-free numpy-idiom coverage for the `graphed.vary` record-time policy m53 adds. The headline
auto-fanout + cross-node resolution + distinctness live in `awkward/m53` and the re-authored
`corpus/m52`; this tree pins the frontend KNOBS (composes_as_union / points= / max_universes), which
are backend-agnostic, so the free-threaded numpy-only CI job covers them.

Fixture: `m53_policy_fixtures.py` — `fanout_weight(**kw)` builds a jes(3) x btag(5) = 15 dependent
weight grid (b-tag SF read off jes-varied pt); `numeric_fanout(**kw)` a numerically-tagged jes
family for a precision coordinate. Extra keywords pass through to `graphed.vary`.

| Work item (plan §) | Test |
|---|---|
| §2.5 composes_as_union collapses to the datacard union | `test_composes_as_union.py::test_composes_as_union_collapses_a_dependent_family_to_the_union` |
| §2.5 composes_as_union + points= is a construction error | `test_composes_as_union.py::test_composes_as_union_with_points_is_a_construction_error` |
| §3 points= PRUNES the grid to the named joints (base untouched) | `test_points_selection.py::test_points_prune_keeps_only_the_named_joints` |
| §3 points= refuses an unreachable point | `test_points_selection.py::test_points_prune_refuses_an_unreachable_point` |
| §3 points= PRECISION accepts a reachable numeric coordinate | `test_points_selection.py::test_points_precision_accepts_a_reachable_numeric_coordinate` |
| §4 the loud guard raises above max_universes, naming count + families | `test_guard.py::test_the_guard_raises_above_the_budget_naming_the_count_and_families` |
| §4 the guard is silent at/below the budget | `test_guard.py::test_the_guard_is_silent_at_the_budget` |
| §4 max_universes= override lets a large grid through | `test_guard.py::test_a_raised_budget_lets_a_grid_through_that_a_low_one_rejects` |
| §4 the default budget admits the benchmark grid | `test_guard.py::test_the_default_budget_admits_the_benchmark_grid` |
| §4 points= / composes_as_union are never guarded | `test_guard.py::test_a_selection_and_a_union_collapse_are_never_guarded` |

## Non-vacuity (fails on the current union-collapse tree for the right reason)
- `points=[list]` → `AttributeError: 'list' object has no attribute 'items'` (m52 expected a Mapping).
- `composes_as_union=` / `max_universes=` → `GraphedError: a variation member must be an Array or a
  Varied, got bool/int` (absorbed as a bogus variation — the keyword does not exist yet).
- guard-count / default-grid tests → `assert '15' in 'got int'` / `assert 7 == 15` (union, not grid).
</content>
