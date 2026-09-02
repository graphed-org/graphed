# `tests/frozen/frontend/m49` — traceability

The m49 non-fill frontend anchors. Awkward-free **by gate**: `ci.yml`'s required `test-freethreaded`
job collects `tests/frozen/frontend` WHOLE on 3.14t with only `pytest hypothesis numpy` installed,
so an `import graphed.awkward` reachable from this tree reds that job at COLLECTION. Helper modules
here carry the `m49_` prefix because the same process also collects `frontend/m48/vary_fixtures.py`
and prepend import mode binds a shared bare name to whichever sibling directory imported first.

## Spellings this tree PINS at m49 freeze (§9.1)

| Surface | Spelling |
|---|---|
| §3.4 impact API | `graphed.impact_by_label(outputs) -> dict[str, tuple[int, ...]]` |
| §5.3 projection stats | `graphed.read_columns_by_label(outputs, source_nid) -> dict[str, tuple[str, ...] \| None]` |
| §2.4 resolution rule | `graphed.member_of(value, label)`, exported out of the package |
| §8.2(i) correspondence | `CompiledGraph.correspondence`, the additive field carrying both halves; this tree reads `.node_map` (`record_node_id -> (reduced_node_id, member_index \| None)`), the per-key frame half (`.frames`) is exercised by `debug/m49` and `checkpoint/m49` |

Both verbs take `Sequence[Varied] | Mapping[str, Sequence[Array]]`; the key set is the §2.4 label
union in §2.4's bound order, resolved by `member_of`, never by the strict `graphed.universe`.

The §5.4 message shape pinned here: the refusing verb's dotted name plus **every** label
`graphed.labels(container)` reports, `"nominal"` included.

The two-form rejection message shape pinned here: a `GraphedError` naming the offending element's
type; for a mapping operand it also names the offending key.

## Test → plan anchor

| Test | Anchor |
|---|---|
| `test_variation_sharing.py::test_second_universe_adds_only_its_own_nodes` | §5.2a — SPAN over one Session vs the no-`vary` ORACLE's own delta |
| `…::test_a_per_universe_prefix_copy_is_visible_in_the_arena_delta` | §5.2a control leg |
| `…::test_shared_prefix_reduces_into_exactly_one_stage` | §5.2c — the prefix in exactly ONE stage |
| `…::test_reduced_stage_count_equals_the_no_vary_oracle` | §5.2c — stage count vs the ORACLE, no frozen literal |
| `…::test_a_per_universe_prefix_copy_is_visible_in_the_reduced_shape` | §5.2c control leg |
| `test_impact_sets.py::test_impact_reports_the_reachability_difference_per_label` | §3.4 — key order, sortedness, empty nominal difference |
| `…::test_a_node_two_labels_share_lands_in_both_impact_sets` | §3.4 — `u ∈ impact(up) ∩ impact(down)`, `u ∉ reachable(nominal)`, `impact(up) ≠ impact(down)` |
| `…::test_the_mapping_form_answers_the_same_impact_sets` | §3.4 — the labelled-mapping operand form |
| `…::test_member_of_is_exported_and_resolves_the_label_union` | §9.1 — `member_of` export, and why `universe` cannot serve |
| `…::test_a_sequence_operand_must_be_all_varied` | §3.4 / §12.4(1) |
| `…::test_a_mapping_operand_must_map_a_label_to_a_sequence_of_array` | §3.4 / §12.4(1) |
| `…::test_a_varied_member_must_not_itself_be_varied` | §3.4 / §12.4(1) — §2.2 nesting the per-label walk cannot resolve |
| `…::test_an_operand_of_neither_form_is_refused` | §3.4 / §12.4(1) — the forms do not mix |
| `test_varied_read_columns_projection.py::test_a_shift_grows_the_union_by_exactly_its_extra_column` | §5.3 — plain union growth, on its own output set |
| `…::test_one_conservative_member_collapses_the_plain_union` | §5.3 / §2.3d — why the growth half needs its own output set |
| `…::test_stats_report_the_shifted_labels_extra_column` | §5.3 — per-label stats, stated order-insensitively |
| `…::test_a_whole_record_consumer_reports_none_not_empty` | §5.3 — `None` ≠ `()` |
| `…::test_the_mapping_form_answers_the_same_stats` | §5.3 — the labelled-mapping operand form |
| `test_boundary_refusal.py::test_join_refuses_a_varied_operand_naming_the_verb_and_the_labels` | §5.4 — both operand positions |
| `…::test_every_boundary_verb_names_the_verb_and_the_labels` | §5.4 — the refusal is the disposition table's class, not one verb |
| `…::test_a_variation_downstream_of_the_join_still_compiles_per_universe` | §5.4 positive control, against a numpy relational reference |
| `test_record_correspondence.py::test_every_surviving_record_id_keys_a_node_of_the_reduced_store` | §8.2(i) — accessor half; the unmarked branch maps to `None` |
| `…::test_the_map_partitions_the_topology_onto_the_reduced_store` | §8.2(i) — the partitioning clause, on the BASE fixture only |
| `…::test_two_labels_sharing_a_node_collapse_onto_one_key` | §8.2(i) — the set-valued key space, via §3.4's verb |
| `…::test_an_identity_token_maps_to_the_node_its_input_landed_in` | §8.2(i) — the post-DCE discriminator |
| `…::test_the_incremental_path_answers_the_same_record_keyed_map` | §8.2(i) — BOTH reduction paths |
| `test_varied_align_path.py` | carried ledger — the varied-mask `vary._align` path (§2.1 one-row-space rule) |
| `test_label_scan_downstream.py` | carried ledger — §2.5's scan is not blind downstream of a container |
| `test_family_tags_through_combination.py` | carried ledger — §1.1 family tags survive a §2.4 combining op |

## Fixture rules this tree obeys

- §5.4's fixture is self-contained: two flat `from_record` tables joined through `graphed.join` on a
  `NumpyBackend`. It does **not** import `frontend/m40/shuffle_backends.py` (awkward + pandas +
  corpus at module scope, and its directory is not on `pythonpath`).
- §5.2a's `CHAIN + 2` expectation is re-derived from the ORACLE's own node-count delta, never
  assumed; §5.2c compares against the ORACLE's stage count, never against §3.3's raw-builder
  literals.
- §8.2(i)'s partitioning clause is asserted on the base fixture (± the unmarked dead branch) only;
  the shared-node extension carries the both-labels clause instead.
- No frozen test here supplies a §7.2 hook: this tree's anchors are the accessor and the frontend
  verbs, never the label association (§8.2(i)).
