# M40 frozen suite — graphed-numpy (relational join half)

Milestone **M40** (distributed joins). M39 shipped the numpy `ShuffleBackend` *exchange* half; M40
adds the *join* half — `match_indices` / `take` / `merge_records` over structured (record) arrays,
`op_form("join")`, the E4 validity carrier, and column projection through both join sides. **Frozen —
read-only after `freeze-M40-0`.** Authoritative contract: `/private/tmp/claude-501/m40-spec.md`
(decisions E1–E5, invariants, traps). These suites run per-directory in pytest prepend mode.

## Files → theme / spec clause

| File | Covers | Spec clause / theme |
|---|---|---|
| `test_join_primitives.py` | `match_indices` (argsort+searchsorted) relational duplication on unsorted/dup keys; `take` gather; `merge_records` field-union-minus-dup-key; inner join bit-for-bit vs a duplicating SQL oracle; determinism; free-fn presence | (a) relational; TARGET 6; §INVARIANTS relational duplication |
| `test_validity_nulls.py` | THE `np.take(-1)` trap: `take(-1)` → invalid not last row; outer-join missing field reads null; mask survives `to_wire`/`from_wire` (E4) and `concat`/`slice_rows` (E4/ADV-r6.3) | (a3) option nulls; TRAP #1; E4 |
| `test_op_form_join.py` | `op_form("join")` = union minus dup key; `how=left/outer` records nullability as companion mask-column fields | (a3) form level; TARGET 7 |
| `test_projection.py` | join reads exactly `{key ∪ used}` on BOTH sides (M5 over-touch, doubled) | (d) projection; TARGET 15 |
| `test_pack_key.py` | GAP-1: direct witness for `graphed.pack_key(array, *, on)` — injectivity, low-field monotonicity, high-field (big-endian) dominance, cross-process/hash-seed determinism, `!= hash()` | TARGET 8; trap #6 |
| `test_grouped_is_awkward_only.py` | GAP-3: the neutral `graphed.join` (the numpy path's entry point) rejects `grouped=True` — that convenience is awkward-only | README pin; §INVARIANTS |

## Discriminating property per test (the WRONG impl each fails)

**test_join_primitives.py**
- `test_match_indices_is_relational_on_unsorted_duplicate_keys` — build key 20 at rows 0 AND 2 on
  UNSORTED input ⇒ two pairs `{(0,0),(2,0)}`. FAILS a grouped/list-of-matches impl (1 pair) and a
  searchsorted impl that assumes pre-sorted input (wrong indices).
- `test_match_indices_inner_drops_unmatched` — inner keeps only the matched pair. FAILS an impl that
  leaks unmatched rows (outer semantics) into `how="inner"`.
- `test_take_gathers_records_by_index` — positional record gather. FAILS an impl that reorders or
  gathers the wrong rows.
- `test_merge_records_is_union_of_fields_minus_duplicate_key` — shared key kept once, both data
  columns present with correct values. FAILS an impl that duplicates the key column or drops a side.
- `test_inner_join_matches_a_duplicating_sql_baseline` — 4 rows (2×2 on key 20) vs an independent
  duplicating oracle, multiset-compared. FAILS any list-of-matches / grouped / deduping join.
- `test_inner_join_is_deterministic` — byte-identical output across two runs. FAILS a nondeterministic
  merge/order.
- `test_join_free_functions_exist_on_the_shuffle_module` — `shuffle.match_indices/take/merge_records`
  exist and agree with the delegate. FAILS an impl that only adds backend methods (the generic engine
  imports the free fns).

**test_validity_nulls.py**
- `test_take_maps_negative_one_to_invalid_not_the_last_row` — index `-1` reads INVALID and its value is
  not the last-row sentinel `999`. FAILS THE `np.take`-based `take` (gathers the last row, leaves the
  row valid) — the row-count-only check it would otherwise pass is explicitly refused here.
- `test_outer_join_missing_field_reads_null_on_each_side` — exactly one null `lx` and one null `rx`,
  asserted on the validity bit. FAILS any impl that fabricates last-row/zero-fill values (zero nulls).
- `test_validity_survives_wire_roundtrip` — the null survives `to_wire`→`from_wire`. FAILS the M39
  `.npy` wire path (no Arrow validity bitmap ⇒ mask dropped on save).
- `test_validity_survives_concat_and_slice_rows` — mask preserved across `concat` then `slice_rows`.
  FAILS plain `np.concatenate` / fancy-slice (drop the mask → all-True, corrupting recombine/spill).

**test_op_form_join.py**
- `test_inner_join_form_is_union_of_fields_minus_duplicate_key` — inner form = `{__joinkey__, lx, rx}`.
  FAILS an impl that duplicates the key or drops a side's fields.
- `test_left_and_outer_forms_record_nullability_as_extra_fields` — left/outer forms have strictly more
  fields than inner (companion mask columns). FAILS a how-ignoring impl (identical fields for
  inner/left) that cannot represent the a3 nulls.

**test_projection.py**
- `test_join_projects_exactly_key_union_used_on_both_sides` — each source's projection is exactly
  `{run, lumi, event} ∪ {its one used column}`. FAILS any impl that over-touches (`lx_unused` /
  `rx_unused` leak into a projected set) on either side.

**test_pack_key.py** (GAP-1 — see `tests/frozen/awkward/m40/test_pack_key.py` for the full rationale
and the measured proof that `hash()` violates monotonicity/dominance)
- `test_pack_key_is_injective_on_distinct_triples` — 12 distinct triples ⇒ 12 distinct `__joinkey__`.
- `test_pack_key_is_strictly_monotonic_in_event_for_fixed_run_lumi` — FAILS `hash()`-based packing.
- `test_pack_key_run_dominates_lumi_and_event` — FAILS `hash()`-based packing.
- `test_pack_key_is_not_pythons_hash` — FAILS `pack_key = lambda t: hash(t)` directly.
- `test_pack_key_is_stable_across_processes_and_hash_seeds` — mirrors the M39 B2 pattern
  (`graphed-exec-check/tests/frozen/m39/test_routing_invariance.py`), adapted to `pack_key`.

**test_grouped_is_awkward_only.py** (GAP-3)
- `test_grouped_is_rejected_on_the_neutral_numpy_join_path` — `graphed.join(..., grouped=True)` must
  raise `TypeError`. FAILS an impl that widens the neutral signature to silently accept `grouped`.

## Non-vacuity (pre-implementation)

`NumpyBackend` has no `match_indices`/`take`/`merge_records` and `op_form` rejects `"join"`;
`graphed` has no `join` or `pack_key`. Every test therefore fails right-reason (AttributeError on the
missing join/pack_key symbols, or op_form raising on the unknown `"join"` op) against current code —
see the verify command in the author's report.
