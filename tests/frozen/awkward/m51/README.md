# m51 frozen suite — `graphed` / `tests/frozen/awkward/m51`

Milestone m51 (variation-aware write-out — skim augmentation). Authority: `systematics-vary-plan.md`
r47 (`8cdf7d4`); §6.4a–g, §9.1's `graphed.selection`, §2.3d's `to_parquet` entry, §7.2 node-id
resolution, §1.1 e-form. Every anchor imports `graphed.awkward`, so none can live in
`tests/frozen/frontend|numpy/m51` (a required free-threaded job collects those under
`pytest hypothesis numpy` alone — awkward's numpy-idiom refusal is the ONLY m51 numpy anchor, and it
is awkward-free). No whole-subtree job collects `tests/frozen/awkward`; `run-tests.sh` runs this dir
in its own process, and the helper carries an `m51_` prefix regardless (prepend import mode).

All m51-new symbols (`select=`, `graphed.awkward.read_varied`, `graphed.selection`) are exercised in
test BODIES, never module-level imports, so the suite COLLECTS against `graphed@main` and fails at run
time for the RIGHT reason (feature absent), per TEST_SANITY.

| anchor | plan clause | test |
|---|---|---|
| `vary-m51-A` | §6.4a superset = level-0 OR, vs an INDEPENDENT eager reference | `test_superset_rows.py::test_written_superset_is_the_union_and_each_universe_is_its_eager_row_set` |
| `vary-m51-B` | §6.4a/c bare-key single-collection Jet skim round-trip (`select={0,1}`) | `test_roundtrip_universes.py::test_bare_key_jet_skim_roundtrips_every_universe` |
| `vary-m51-B` | §6.4a/c/d field-scoped `{Jet,MET}` object-migration round-trip (`select={0,("Jet",1)}`) | `test_roundtrip_universes.py::test_field_scoped_multifield_skim_roundtrips_every_universe` |
| `vary-m51-B` | §7.2 all-zero-delta REPLICATION + weight-only storable + e-canonical `murf_5em1` | `test_roundtrip_universes.py::test_weight_only_labels_replicate_the_collapsed_output_and_roundtrip` |
| `vary-m51-C` | §9.1 `graphed.selection` on a ROOT context is `None` | `test_selection_bridge.py::test_selection_on_a_root_context_is_none` |
| `vary-m51-C` | §6.4a bridge write ≡ hand-written mask | `test_selection_bridge.py::test_bridge_write_roundtrips_identically_to_the_hand_written_mask` |
| `vary-m51-C` | §9.1 case: skip `vary` identity links | `test_selection_bridge.py::test_bridge_walks_vary_identity_links_and_roundtrips_the_prevary_spelling` |
| `vary-m51-C` | §6.4a(2a) vary-link admission discriminator (record `E1`, mask `E2`) | `test_selection_bridge.py::test_bridge_admits_a_vary_link_between_record_and_mask_handles` |
| `vary-m51-C` | §6.4a(2a) handle equality, not object `is` (re-recorded mask) | `test_selection_bridge.py::test_bridge_accepts_a_re_recorded_equal_mask_expression` |
| `vary-m51-C` | §9.1 case-2 universe/nominal → grandparent Array + (2a) refuse downstream | `test_selection_bridge.py::test_selection_on_a_universe_nominal_context_is_a_grandparent_array_and_refuses_downstream` |
| `vary-m51-D` | §6.4a(1) multiplicity offsets — EXECUTION-time, not record-time | `test_entry_checks.py::test_multiplicity_change_is_refused_at_execution_time_not_record_time` |
| `vary-m51-D` | §6.4a(2a) silent-corruption lineage refuse | `test_entry_checks.py::test_2a_refuses_a_record_whose_context_the_mask_does_not_derive_from` |
| `vary-m51-D` | §6.4a(2a) chained-context refuse | `test_entry_checks.py::test_2a_refuses_a_chained_context_mask_against_a_root_row_space_record` |
| `vary-m51-D` | §6.4a(2a·i) context-free record SKIPS 2a (positive) | `test_entry_checks.py::test_2a_is_skipped_for_a_context_free_record` |
| `vary-m51-D` | §6.4a(2a·ii) contexted record + handleless mask refuse | `test_entry_checks.py::test_2a_refuses_a_contexted_record_with_a_handleless_mask` |
| `vary-m51-D` | §2.3e ORIGINATION positive (mask through record's context) | `test_entry_checks.py::test_2a_accepts_a_mask_that_originates_the_records_handle` |
| `vary-m51-D` | §6.4a(2c) jagged level-0 refuse — DEPTH alone (flat positive) | `test_entry_checks.py::test_2c_refuses_a_jagged_level_0_mask_that_passes_2a_and_2b` |
| `vary-m51-D` | §6.4a(2c) too-shallow level-1 mirror refuse | `test_entry_checks.py::test_2c_refuses_a_too_shallow_mask_supplied_at_level_1` |
| `vary-m51-D` | §6.4a bare-key ambiguity (two jagged fields) + field-scoped positive | `test_entry_checks.py::test_bare_depth_key_on_two_independently_jagged_fields_is_ambiguous` |
| `vary-m51-D` | §6.4b row-space refusal — blamed row-space, NOT offsets | `test_entry_checks.py::test_6_4b_refuses_a_selection_scoped_stored_weight_naming_the_row_space` |
| `vary-m51-D` | §6.4b vary-reached stored weight (positive) | `test_entry_checks.py::test_6_4b_accepts_a_vary_reached_stored_weight` |
| `vary-m51-E` | §6.4b/c value=xor / mask=packbits under the bound names | `test_representation.py::test_value_deltas_are_xor_and_masks_are_packbits_under_the_bound_names` |
| `vary-m51-E` | §6.4b nested `Jet.pt` flattens to `Jet_pt`, reads back via manifest | `test_representation.py::test_nested_field_path_flattens_and_reads_back_through_the_manifest` |
| `vary-m51-E` | §6.4b collision (nested `Jet.pt` + flat `Jet_pt`) refused naming both | `test_representation.py::test_collision_between_a_nested_and_a_flat_field_is_refused_naming_both` |
| `vary-m51-E` | §6.4b non-varying `Jet_pt` beside varied `Jet.pt` is NOT a collision | `test_representation.py::test_a_nonvarying_flat_field_is_not_a_collision` |
| `vary-m51-F` | §6.4e byte-identical manifest across two `PYTHONHASHSEED` processes | `test_manifest_determinism.py::test_manifest_bytes_are_identical_across_two_hash_seeds` |
| `vary-m51-F` | §6.4e sorted mapping keys + levels-list bound order | `test_manifest_determinism.py::test_manifest_mapping_keys_are_sorted_and_levels_list_is_ordered` |
| `vary-m51-G` | §6.4e object-migration manifest KEY SET + `levels == [0,["Jet",1]]` | `test_manifest.py::test_object_migration_manifest_key_set_and_levels_list` |
| `vary-m51-G` | §6.4e weight-only KEY SET + `levels == [0]` | `test_manifest.py::test_weight_only_manifest_key_set_and_levels_list` |
| `vary-m51-G` | §6.4e augmented file round-trips through `ak.from_parquet` | `test_manifest.py::test_augmented_file_roundtrips_through_ak_from_parquet` |
| `vary-m51-G` | §6.4g unvaried write keeps `ak.to_parquet` bytes, NO manifest (same-process) | `test_manifest.py::test_unvaried_write_keeps_ak_to_parquet_bytes_and_writes_no_manifest` |
| `vary-m51-H` | §6.4d structure refusal naming label+field (exec-time) | `test_structure_refusal.py::test_multiplicity_changing_field_is_refused_naming_the_label_and_field` |
| `vary-m51-H` | §6.4d same-multiplicity object migration writes+round-trips (positive) | `test_structure_refusal.py::test_same_multiplicity_object_migration_still_writes_and_roundtrips` |
| `vary-m51-I` | §6.4f/§7.2 write-path optimizer-merge refusal (`w*1.0`) — record-time | `test_optmerge_refusal.py::test_optimizer_mergeable_label_is_refused_at_the_call` |
| `vary-m51-I` | §6.3 unvaried write path unchanged (positive) | `test_optmerge_refusal.py::test_unvaried_write_path_is_unchanged` |
| `vary-m51-J` | §2.3d `to_parquet` *accepting*: Varied record + Varied select consumed, returns paths | `test_to_parquet_disposition.py::test_a_varied_record_and_varied_select_are_consumed_and_return_paths` |
| `vary-m51-N` | §5.2b/§7.1 augmented write is a single read pass (P reads, not P·labels) | `test_single_read.py::test_a_varied_write_reads_each_partition_exactly_once` |

## Spellings this freeze pins (the plan defers these to m51 freeze — §6.4a/b/e, §9.1)

* **`graphed.selection(ctx)`** — top-level module export; the §9.1 three-case bridge verb.
* **`graphed.awkward.read_varied(path) -> {label: ak.Array}`** — the symmetric awkward-idiom reader.
* **`select=`** — a single `Varied` row mask (⇔ `{0: mask}`), OR a mapping keyed by `int` (bare
  depth, the RECORD'S OWN axis) or `tuple[str, int]` = `(field_name, depth≥1)` (field-scoped).
* **Value-delta column** `__vary_{label}__{field_flat}` (`__vary_murf_5em1__Jet_pt`); the field path
  is flattened per level with `_` (`Jet.pt`→`Jet_pt`; a bare-field skim's `pt`→`pt`).
* **Validity-mask column** `__vary_{label}__mask__{entry_flat}` (`…__mask__0`, `…__mask__1`,
  `…__mask__Jet_1`). (The round-trip resolves THROUGH the manifest; only the value-delta literal and
  the identifier-shape/label-verbatim property are asserted for masks — E.)
* **Parquet KV key** `b"graphed.variations"`.
* **Manifest** (json under that key): top-level keys are every label (`"nominal"` included) PLUS the
  reserved `"levels"`. `manifest[label] = {column: {"representation": "base"|"xor"|"packbits",
  "field": <flat_field>|null, "entry": <int | [flat_field, depth]>|null}}`. `manifest["levels"]` is
  a LIST of `int` or `[flat_field, depth]`, ordered by `(depth, flat_field or "")`. Serialized
  `json.dumps(manifest, sort_keys=True)` (sorted mapping keys, `PYTHONHASHSEED`-free; list order is
  the explicit key — `sorted()` cannot order the mixed list and `sort_keys` never reorders a list).
* **Reconstruction contract** (§6.4c): `read_varied(path)[L]` is the record for universe L, restricted
  to the rows passing L's stored level-0 mask AND, at every supplied level ≥ 1, the objects passing
  L's stored mask — bit-exact vs the in-memory varied run.

## Notes for the implementer

* **RECORD-time vs EXECUTION-time is frozen by the compute=False witness** (trap ledger §7):
  record-time predicates (2a lineage, 2c depth, bare-key ambiguity, §6.4f optimizer-merge, §6.4b
  row-space) raise at the `to_parquet` CALL, so `compute=False` still raises; execution-time ones
  ((1) multiplicity offsets, (2b) row-count, level-≥1 structural) return a `Plan` from `compute=False`
  and raise only when it RUNS. Do NOT move a predicate across this line.
* **`to_parquet`'s *accepting* disposition is BEHAVIORAL, not table-registered.** The frozen m48 gate
  `awkward/m48/test_module_verb_dispositions.py::test_to_parquet_carries_no_disposition_until_m51`
  hard-asserts `to_parquet` stays OUT of `VERB_DISPOSITIONS` and out of the discovered surface (keep
  its first parameter annotated `Any`). Prove accepting by calling it (anchor J), never by a table row.
* **§7.2 node-id resolution is load-bearing for anchor B's weight-only round-trip.** The `murf`
  family's `murf_1` member interns to nominal's node id, so `mark_output` de-dups it and the value
  list is shorter than the marked-output list; a POSITIONAL unpack misassigns `murf_5em1`/`murf_2`.
  The test witnesses the collapse (node-id equality) AND the correct per-label reconstruction.
* **`graphed.selection` is NOT a thin wrapper over `_selection()`** (returns `None` on a project
  link). Case-2 must return the label's member of the parent's selection — anchor C's
  universe/nominal half fails until the real case-2 walk exists.
* The no-`awkward._connect`-import rule (§6.4e) is UNANCHORED here by design — it rides code review +
  the integrity scan; discharge it with a one-line `tests/extra` static assertion if desired.

## Non-vacuity (TEST_SANITY: 35 failed / 2 passed at freeze)

Every test fails against `graphed@main` for the RIGHT reason — `select=` unknown-keyword `TypeError`,
`graphed.selection` `AttributeError`, or (numpy tree) the missing awkward-naming guard — never an
import error or a fixture bug. The only two green tests are the UNVARIED positive controls
(`test_unvaried_write_keeps_ak_to_parquet_bytes…`, `test_unvaried_write_path_is_unchanged`), which
pin that the pre-selection path is unchanged; each of their files carries red siblings.
