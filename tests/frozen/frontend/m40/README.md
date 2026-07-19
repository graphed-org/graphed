# M40 frozen suite — graphed frontend (distributed joins)

Milestone **M40**. graphed owns the backend-agnostic join frontend: the neutral `graphed.join`
module verb, the multi-source `graphed.join_plan` builder, and the determinism/backend-independence
guarantees of both. Themes covered here: **(a) relational bit-for-bit**, **(a2) backend-independence**,
and the **determinism gate**. The a3 (option nulls), a4 (grouped), projection, broadcast, skew, and
bounded-memory themes live in the `awkward`/`numpy`/executor m40 suites. **Frozen — read-only.**

## Files → contract clause / theme

| File | Covers | Contract row |
|---|---|---|
| `shuffle_backends.py` | fixtures: `REAL_BACKENDS` (AwkwardBackend + NumpyBackend cases: source ctor + row normalizer), the fixed corpus skim -> two duplicating/orphan tables, the duplicating pandas reference, and a toy `ShuffleBackend`+`ListSource` for the plan-shape test | §3.1-§3.3 |
| `test_relational_join.py` | (a)/(a2): `graphed.join(on, how="inner")` materialized on BOTH backends is bit-for-bit the duplicating pandas merge; duplication vs list-of-matches; inner drops orphans; flat union-minus-key columns | (a), (a2) |
| `test_join_plan_builder.py` | `graphed.join_plan(...)` emits two `map_write` stages + one `gather_join` (`inputs=(0,1)`); byte-identical `to_bytes()` across runs | E1 / target 13 |
| `test_determinism.py` | identical input -> byte-identical compiled join IR across two runs (both backends); recorded join graph structurally identical across awkward and numpy | determinism gate |

## Pinned surface (test-author decisions, from the resolved contract)

- `graphed.join(left, right, *, on, how="inner") -> Array` — module verb (idiom-neutral, NOT an
  `Array` method); both inputs share one `Session`; the result is a flat relational record whose
  fields are the union of both sides minus the duplicated key. Inner-join duplication is SQL/pandas
  semantics (k left * m right matches => k*m rows). Materialized via `Session.materialize`.
- `graphed.join_plan(output, *, backend=None, steps_per_file=1) -> DurablePlanV2` — mirrors
  `shuffle_plan`; stage `kind`s are two `"map_write"` + one `"gather_join"` with `inputs=(0, 1)`.
- Both are added to `graphed.__all__`; referenced via attribute access (`graphed.join`) so the suite
  COLLECTS today and fails per-test with `AttributeError` until the symbols exist.

## Non-vacuity / discrimination (the wrong impl each test kills)

| Test | Mechanism it witnesses | Wrong impl it FAILS |
|---|---|---|
| `test_graphed_join_is_a_neutral_module_verb` | `graphed.join` is callable and absent from `Array` | join hung off `Array` (idiom leak); no join verb at all |
| `test_inner_join_is_bit_for_bit_the_duplicating_pandas_merge` [awkward,numpy] | materialized rows == duplicating pandas merge (row multiset), on both backends | any join whose result differs from SQL merge; an awkward-only primitive that fails on numpy (a2) |
| `test_join_duplicates_matches_not_list_of_matches` [awkward,numpy] | 10 rows (3·2+2·1+1·2) across 3 keys; `rows > distinct_keys` | a list-of-matches / grouped baseline (≤6 nested rows) — contract trap 3 |
| `test_inner_join_drops_unmatched_orphan_keys` [awkward,numpy] | left-only 88 and right-only 99 absent; keys == {10,20,30} | a left/outer impl, or one ignoring `how=`, that keeps orphans |
| `test_output_columns_are_the_union_minus_duplicate_key` [awkward,numpy] | every output row has all of `run,lumi,event,njet,nmu` | a nested/grouped output (unreadable row-wise); dropping a payload column |
| `test_join_plan_emits_two_map_writes_and_a_gather_join` | plan has exactly two `map_write` + a `gather_join` | a single-`map_write` (single-source) plan; fusing the join into one stage |
| `test_gather_join_depends_on_both_map_write_stages` | `gather_join.inputs` ⊇ both map-write indices | a plan missing the two-input barrier edge (one side dropped) |
| `test_join_plan_to_bytes_is_deterministic_across_runs` | byte-identical `DurablePlanV2.to_bytes()` across builds | nondeterministic routing / task-id folding |
| `test_recorded_join_ir_is_byte_identical_across_runs` [awkward,numpy] | identical input -> byte-identical compiled IR | any nondeterminism in interning/serialization of the join node |
| `test_recorded_join_graph_is_backend_independent` | `to_dot()` identical across awkward and numpy | a join that records backend-specific structure into the graph |

## Corpus usage

Payload columns (`njet`, `nmu`) are derived from a FIXED skim
`graphed_corpus.make_events(n_events=24, seed=1234)` (the M3/M5 fixture; on `pythonpath` via
`tests/_corpus`). The corpus records carry no event id, so `(run, lumi, event)` keys are synthesized
with controlled multiplicity and orphans (see `skim_tables`) so the duplicating inner-join semantics
are actually exercised and hand-verifiable.
