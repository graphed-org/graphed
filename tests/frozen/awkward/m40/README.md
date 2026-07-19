# M40 frozen suite — graphed-awkward (distributed-join primitives)

Milestone **M40**. `AwkwardBackend` gains the `ShuffleBackend` **join half**
(`match_indices`/`take`/`merge_records`), `op_form("join")` learns the flat relational
record-merge, and `gak.join` adds the awkward-only grouped convenience + column projection through
the join. **Frozen — read-only after `freeze-M40-0`.** Themes covered here: (a) relational
primitives, `op_form("join")`, (a3) option nulls, (a4) grouped, (d) projection. (Themes serialize/
ir live in `core/m40`; bit-for-bit-vs-pandas + backend-independence in `frontend/m40`; broadcast/
cost/skew/B5/arrival in the executors `m40` suite.)

## Files → theme / plan clause

| File | Covers | Plan clause / theme |
|---|---|---|
| `test_join_primitives.py` | relational join primitives, `merge_records` field-union, (a3) option nulls, `op_form("join")` merge form | (a), (a3); §3.0, §3.3, §2.5 |
| `test_grouped.py` | (a4) `gak.join(grouped=True)` = relational regrouped by deterministic `ak.unflatten` | (a4); §3.1, §3.3 |
| `test_projection.py` | (d) join reads exactly `{key ∪ used}` on each side (M5 over-touch, two sources) | (d); §3.4 |
| `test_pack_key.py` | GAP-1: direct witness for `graphed.pack_key(array, *, on)` — the thin neutral frontend verb (mirrors `graphed.join`) that builds the flat u64 `__joinkey__` column `graphed.join` reuses internally. Pins injectivity, low-field monotonicity, high-field (big-endian) dominance, cross-process/hash-seed determinism, and `!= hash()` | target 8; trap #6 |
| `test_join_routing_golden.py` | GAP-2: `graphed.pack_key(run=0,lumi=0,event=e)` must equal `e` exactly, and those keys must route per the frozen M39 `golden_route.GOLDEN` table via the shared `be.partition` | INVARIANTS ("join routing must satisfy them too") |

## Traceability — each test's discriminating property

| Test | Witnessed mechanism | Wrong impl it FAILS (discriminator) |
|---|---|---|
| `test_inner_join_primitives_duplicate_relationally` | k=2 build × k=2 probe on one key ⇒ **4** output rows; only the matched key survives; fields = union − dup key | a **list-of-matches / grouped / first-match** join (2 rows) — the §3.3 relational-duplication pin (trap 3) |
| `test_merge_records_is_field_union_minus_duplicate_key` | `merge_records` zips two equal-length gathered blocks column-wise, dropping the ONE shared key | a **keep-both-keys** merge (two keys / suffixed key ⇒ field set differs) |
| `test_left_join_missing_side_reads_as_an_awkward_option_null` | `how="left"` keeps every left row; the unmatched row's right field reads **`ak.is_none` True** (validity), matched row reads its value | **trap 1** — `take` via `np.take(-1)` or naive `block[index]` (index −1 → negative-index the LAST row) returns a real value ⇒ `is_none` False; a row-count-only check would pass it |
| `test_op_form_join_is_a_flat_record_merge_union_minus_key` | `op_form("join")` output form = union of both sides' fields minus dup key; inner introduces no option | an **identity/passthrough** `op_form` (returns `inputs[0]` ⇒ only `{__joinkey__, lv}`) |
| `test_op_form_left_join_makes_the_outer_side_option_typed` | `how="left"` ⇒ the missing-side field is **option-typed** (`?` in the type string) | an impl that ignores `how` and builds a non-option merge |
| `test_grouped_join_is_the_relational_result_regrouped_by_unflatten` | `grouped=True` ⇒ `len` = matching **build rows** (2), `ak.num` = match counts (2,2), and `ak.flatten` reproduces the relational multiset | an impl returning the **ungrouped** relational result for `grouped=True` (`len == 4`, `num` fails); a nondeterministic / wrong-count grouping |
| `test_join_reads_exactly_key_plus_used_on_the_left` | left read set = `{run,lumi,event,lv}` exactly; `lx` absent | Join arm that **over-touches** (gathers the full merged record on the reporting typetracer ⇒ `lx` read) |
| `test_join_reads_exactly_key_plus_used_on_the_right` | right read set = `{run,lumi,event,rv}` exactly; `rx` absent | same over-touch on the right side |
| `test_neither_side_pulls_the_other_sides_columns` | left read set has no `rv/rx`; right has no `lv/lx` | a join that reads both sides' full schema on each partition (the two-source over-touch bug) |
| `test_pack_key_is_injective_on_distinct_triples` | 12 distinct `(run,lumi,event)` ⇒ 12 distinct `__joinkey__` | a packer that collapses fields (keys only on one field) |
| `test_pack_key_is_strictly_monotonic_in_event_for_fixed_run_lumi` | fixed `(run,lumi)`, ascending `event` ⇒ strictly increasing `__joinkey__` | `hash()`-based packing (measured non-monotonic in the module docstring) |
| `test_pack_key_run_dominates_lumi_and_event` | every `run=2` key > every `run=1` key over the tested `lumi`/`event` range | `hash()`-based packing (measured non-dominant in the module docstring) |
| `test_pack_key_is_not_pythons_hash` | `__joinkey__ != hash(triple)` and `!= hash(field bytes)` | `pack_key = lambda t: hash(t)` directly |
| `test_pack_key_is_stable_across_processes_and_hash_seeds` | identical across two child processes launched with different `PYTHONHASHSEED` | nondeterminism leaking from dict/set order or id()-based folding |
| `test_pack_key_zero_high_fields_routes_per_the_m39_golden_vectors` | `pack_key(0,0,e) == e` exactly, and those keys route per frozen `GOLDEN` via `be.partition` | a join co-partitioning path that reimplements routing instead of reusing `ShuffleBackend.partition` |

## Non-vacuity (measured against current HEAD, `.venv-m40`)

Pre-implementation the M40 join API does not exist, so every test fails for the right reason:

```
$ cd tests/frozen/awkward/m40 && python -m pytest -q
15 failed
  match_indices missing (2) · merge_records missing (1) · op_form("join") → TypeError
  "unsupported awkward op 'join'" (2) · gak.join missing (4) · graphed.pack_key missing (6)
```

Discriminating designs (not merely "API missing"): the relational tests assert the **duplicated row
count / multiset**, so a syntactically-present but list-of-matches/grouped/first-match join still
fails; the (a3) test asserts **`ak.is_none` validity** (not row count), so a `take` that negative-
indexes or `np.take(-1)`s a real value fails; the projection tests pin the **exact** read set with an
explicit unused-column exclusion, so an over-touching Join arm fails even once `gak.join` exists.
