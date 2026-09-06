# m48 frozen suite — `graphed` / `tests/frozen/awkward`

Milestone m48 (`vary` frontend + weight path). Authority: `systematics-vary-plan.md` r33.
Every anchor here imports `graphed.awkward` — the gak enumerations and representatives, the whole
§2.6 event-context family, and §4.1's correctionlib recording — so it cannot live in
`tests/frozen/frontend/m48`, which a required free-threaded CI job collects under
`pytest hypothesis numpy` alone (§10/m48 partition rule (2)). No whole-subtree job collects
`tests/frozen/awkward`; `scripts/run-tests.sh` runs this directory in its own process.

| anchor | plan clause | test |
|---|---|---|
| `vary-m48-A1` | §1.1 kwarg channel, exact-decimal normalization | `test_tag_grammar.py::test_kwarg_tags_canonicalize_by_exact_decimal_arithmetic` |
| `vary-m48-A1` | §1.1 `points=` channel; PDF indices untouched | `test_tag_grammar.py::test_the_variations_channel_carries_tags_kwarg_syntax_cannot_spell` |
| `vary-m48-A1` | §1.1 collection-mapping inner keys (third channel) | `test_tag_grammar.py::test_the_collection_mapping_inner_keys_are_the_third_channel` |
| `vary-m48-A1` | §1.1 non-minimal canonical tags re-rendered | `test_tag_grammar.py::test_non_minimal_canonical_grammar_tags_are_re_rendered` |
| `vary-m48-A1` | §1.1 unification ACROSS calls | `test_tag_grammar.py::test_two_spellings_of_one_value_unify_ACROSS_calls` |
| `vary-m48-A1` | §1.1 duplicate-after-canonicalization WITHIN one call | `test_tag_grammar.py::test_two_spellings_of_one_value_are_a_duplicate_rejection_WITHIN_one_call` |
| `vary-m48-A1` | §1.1 cross-notation pairs, one call | `test_tag_grammar.py::test_cross_notation_numeric_equal_pairs_are_rejected_within_one_call` |
| `vary-m48-A1` | §1.1 cross-notation pairs across stacking weight calls | `test_tag_grammar.py::test_cross_notation_pairs_are_rejected_across_two_STACKING_calls` |
| `vary-m48-A1` | §1.1 duplicate inside one collection mapping | `test_tag_grammar.py::test_duplicate_after_canonicalization_inside_one_collection_mapping` |
| `vary-m48-A1` | §1.1 malformed / non-string tag rejections | `test_tag_grammar.py::test_malformed_and_non_string_tags_are_rejected` |
| `vary-m48-A1` | §1.1 negative zero | `test_tag_grammar.py::test_negative_zero_canonicalizes_to_zero` |
| `vary-m48-A1` | §1.1 cap boundary pair, refused BY CAUSE | `test_tag_grammar.py::test_the_cap_boundary_pair_refuses_by_cause` |
| `vary-m48-A1` | §1.1 integer-magnitude rejection as a distinct case | `test_tag_grammar.py::test_an_integer_magnitude_over_the_cap_names_the_magnitude` |
| `vary-m48-A1` | §2.1 signature-shadowed names incl. self-reference | `test_tag_grammar.py::test_signature_shadowed_names_reach_tags_through_the_mapping_channels` |
| `vary-m48-A1` | §1.1 the tag `nominal` is legal | `test_tag_grammar.py::test_the_tag_nominal_is_legal_and_yields_an_ordinary_label` |
| `vary-m48-A1` | §2.1 `points=` refused in the shift form | `test_tag_grammar.py::test_variations_is_refused_in_the_shift_form` |
| `vary-m48-A1` | §2.1 `nominal=` refused, naming `collections=` | `test_tag_grammar.py::test_nominal_is_refused_in_the_shift_form_with_an_error_naming_collections` |
| `vary-m48-A1` | §1.1 no label contains `.` or `-` | `test_tag_grammar.py::test_no_label_ever_contains_a_dot_or_a_dash` |
| `vary-m48-A1` | §1.1 tag given twice / empty tag set | `test_tag_grammar.py::test_a_tag_given_twice_and_an_empty_tag_set_are_rejected` |
| `vary-m48-A2` | §2.1 stacking base case on a LOOSE `Varied` | `test_vary_stacking.py::test_stacking_on_a_loose_varied_keeps_the_members_idiom_and_the_inherited_universes` |
| `vary-m48-A2` | §2.1(b) two-level `old_ambient[L] x factor[L]` | `test_vary_stacking.py::test_a_weight_vary_composes_with_the_inherited_shift_label_two_levels_deep` |
| `vary-m48-A2` | §2.1(b) ROW-SPACE positive control | `test_vary_stacking.py::test_a_factor_read_at_the_parent_is_accepted_and_re_indexed_to_the_derived_row_space` |
| `vary-m48-A2` | §2.1(b) DESCENDANT negative control | `test_vary_stacking.py::test_a_factor_read_through_a_DESCENDANT_context_is_a_construction_time_error` |
| `vary-m48-A3` | §2.3d exhaustive disposition over the union | `test_module_verb_dispositions.py::test_every_discovered_verb_carries_a_disposition` |
| `vary-m48-A3` | §2.3c/§2.3d floor over the UNION of three enumerations | `test_module_verb_dispositions.py::test_the_floor_is_asserted_over_the_union_of_the_three_enumerations` |
| `vary-m48-A3` | §2.3d `evaluate_ir` exclusion | `test_module_verb_dispositions.py::test_evaluate_ir_is_outside_the_array_consuming_surface` |
| `vary-m48-A3` | §2.3d `to_parquet` freeze-order fence (m51) | `test_module_verb_dispositions.py::test_to_parquet_carries_no_disposition_until_m51` |
| `vary-m48-A3` | §2.3d idiom-package classifications | `test_module_verb_dispositions.py::test_the_idiom_packages_classifications_are_bound_per_verb` |
| `vary-m48-A3` | §2.3d refusal contract one (boundary/plan verbs) | `test_module_verb_dispositions.py::test_the_boundary_and_plan_verbs_refuse_without_silently_compiling` |
| `vary-m48-A3` | §2.3d refusal contract two (compile/aggregate verbs) | `test_module_verb_dispositions.py::test_the_compile_and_aggregate_verbs_refuse_naming_graphed_universe` |
| `vary-m48-A3` | §2.3d expanding verbs, per verb | `test_module_verb_dispositions.py::test_the_expanding_verbs_are_asserted_PER_VERB` |
| `vary-m48-A3` | §2.3d idiom expanding verbs' own return types | `test_module_verb_dispositions.py::test_the_idiom_expanding_verbs_return_their_own_type_per_label` |
| `vary-m48-A3` | §2.3d `reindex_to` / `unify_contexts` dispositions | `test_module_verb_dispositions.py::test_reindex_to_broadcasts_and_unify_contexts_carries_no_disposition` |
| `vary-m48-A3` | §2.2 reserved names + string-getitem control | `test_module_verb_dispositions.py::test_node_id_and_session_raise_while_the_field_of_that_name_still_reads` |
| `vary-m48-A3` | §2.2 property half classified by measurement | `test_module_verb_dispositions.py::test_the_property_half_is_classified_by_measurement` |
| `vary-m48-A3` | §2.3e `context_of` on a container = most-derived | `test_module_verb_dispositions.py::test_context_of_on_a_container_answers_with_the_MOST_DERIVED_handle` |
| `vary-m48-A4` | §2.3c the binding discovery rule's own premises | `test_gak_classification.py::test_the_discovery_rule_reads_the_module_gak_actually_is` |
| `vary-m48-A4` | §2.3c exhaustive gak classification | `test_gak_classification.py::test_every_discovered_gak_function_carries_a_classification` |
| `vary-m48-A4` | §2.3c non-vacuity floor | `test_gak_classification.py::test_the_non_vacuity_floor` |
| `vary-m48-A5` | §2.3e(2) handle propagation over the carrying classes | `test_context_handle_propagation.py::test_every_carrying_function_propagates_the_context_handle_it_was_given` |
| `vary-m48-A5` | §2.3e(2) `src`-side argument fixtures exist | `test_context_handle_propagation.py::test_every_carrying_function_has_an_argument_fixture_in_src` |
| `vary-m48-A5` | §2.3e(3) exempt set is exactly two classes | `test_context_handle_propagation.py::test_the_exempt_set_is_exactly_eager_metadata_and_refusing` |
| `vary-m48-A5` | §2.3e(3) membership floor | `test_context_handle_propagation.py::test_the_membership_floor_on_the_exempt_classes` |
| `vary-m48-A6` | §2.3c *container-traversing* representative | `test_gak_variation_behaviour.py::test_zip_traverses_its_mapping_argument` |
| `vary-m48-A6` | §2.3c *tuple-returning* representative | `test_gak_variation_behaviour.py::test_unzip_returns_a_tuple_of_varied` |
| `vary-m48-A6` | §2.3c *eager-metadata* representatives | `test_gak_variation_behaviour.py::test_the_eager_metadata_verbs_answer_on_the_nominal_member` |
| `vary-m48-A6` | §2.4 Varied-meets-itself alignment | `test_gak_variation_behaviour.py::test_a_varied_meeting_one_derived_from_itself_stays_label_aligned` |
| `vary-m48-A6` | §2.4 bound union ORDER | `test_gak_variation_behaviour.py::test_the_union_order_is_the_first_operands_then_the_seconds` |
| `vary-m48-A6` | §2.4 absent label falls back to nominal (no cross products) | `test_gak_variation_behaviour.py::test_a_label_absent_from_one_operand_takes_that_operands_nominal_member` |
| `vary-m48-A6` | §2.5 unknown label lists the valid ones | `test_gak_variation_behaviour.py::test_an_unknown_label_raises_listing_the_valid_ones` |
| `vary-m48-A6` | §2.5 form-incompatible member names the label | `test_gak_variation_behaviour.py::test_a_form_incompatible_member_is_a_construction_time_error_naming_the_label` |
| `vary-m48-A6` | §2.1 cross-Session / cross-source members refused | `test_gak_variation_behaviour.py::test_members_from_another_session_or_another_source_are_refused` |
| `vary-m48-A7` | §2.6b `vary` returns a NEW context; input unchanged | `test_event_context_lineage.py::test_vary_returns_a_new_context_and_the_input_is_unchanged` |
| `vary-m48-A7` | §2.3e context-handle ORIGINATION | `test_event_context_lineage.py::test_the_same_read_through_a_derived_context_and_its_parent_has_one_node_id_and_two_handles` |
| `vary-m48-A7` | §2.3e ancestor chain unifies to the most derived | `test_event_context_lineage.py::test_ancestor_chain_inputs_unify_to_the_most_derived_context` |
| `vary-m48-A7` | §2.3e divergence AT THE OP, naming both | `test_event_context_lineage.py::test_divergent_contexts_raise_AT_THE_OP_naming_both` |
| `vary-m48-A7` | §2.6b pure derivations are canonical | `test_event_context_lineage.py::test_pure_derivations_are_canonical_so_two_reads_of_one_universe_unify` |
| `vary-m48-A7` | §2.1 divergence at `vary`'s own construction | `test_event_context_lineage.py::test_divergence_is_also_caught_at_varys_OWN_construction` |
| `vary-m48-A7` | §2.2 `labels(ctx)` reports the shift labels | `test_event_context_lineage.py::test_labels_on_a_context_reports_the_shift_labels_it_carries` |
| `vary-m48-A7` | §2.2 `universe`/`nominal` return a CHILD context | `test_event_context_lineage.py::test_universe_and_nominal_return_a_context_that_is_a_CHILD_of_the_argument` |
| `vary-m48-A8` | §2.2 union terms (a) and (c) | `test_event_context_labels.py::test_term_a_and_term_c_together_the_union_is_not_the_masks_labels_alone` |
| `vary-m48-A8` | §2.2 union term (b) alone | `test_event_context_labels.py::test_term_b_alone_a_shift_varied_collection_with_an_UNVARIED_derivation_mask` |
| `vary-m48-A8` | §2.2 union term (c) alone | `test_event_context_labels.py::test_term_c_alone_a_mask_varied_through_the_LOOSE_primitive` |
| `vary-m48-A9` | §2.6d data-context guard, both forms | `test_event_context_guards.py::test_a_data_context_refuses_BOTH_vary_forms` |
| `vary-m48-A9` | §2.6a lockstep shared-tag-set validation | `test_event_context_guards.py::test_lockstep_collections_must_share_one_tag_set` |
| `vary-m48-A9` | §2.6a no reserved names | `test_event_context_guards.py::test_the_context_reserves_no_names` |
| `vary-m48-A9` | §2.6a slice/int subscript refusal + controls | `test_event_context_guards.py::test_slice_and_int_subscripts_are_refused_naming_the_supported_forms` |
| `vary-m48-A9` | §2.2 the three verbs on both input shapes | `test_event_context_guards.py::test_universe_labels_and_nominal_answer_on_both_a_Varied_and_a_context` |
| `vary-m48-A10` | §2.6c selection-scoped weight; parent untouched | `test_ambient_weight_registry.py::test_a_selection_scoped_weight_leaves_the_parent_untouched` |
| `vary-m48-A10` | §2.6c per-label re-indexing, elementwise | `test_ambient_weight_registry.py::test_the_derived_registry_is_re_indexed_per_label_elementwise` |
| `vary-m48-A10` | §2.6c the wrong-mask answer differs (instrument) | `test_ambient_weight_registry.py::test_re_indexing_every_label_by_nominals_mask_is_a_different_answer` |
| `vary-m48-A10` | §2.1(b) weight answers in the context's row space | `test_ambient_weight_registry.py::test_the_ambient_weight_answers_in_the_contexts_OWN_row_space` |
| `vary-m48-A11` | §6.1d(A) most-derived on one chain | `test_lineage_seams.py::test_unify_contexts_answers_the_most_derived_handle_on_one_chain` |
| `vary-m48-A11` | §6.1d(A) `None` when all context-free | `test_lineage_seams.py::test_unify_contexts_is_None_when_every_argument_is_context_free` |
| `vary-m48-A11` | §6.1d(A) adopt rule | `test_lineage_seams.py::test_unify_contexts_ignores_context_free_arguments_beside_contexted_ones` |
| `vary-m48-A11` | §6.1d(A) divergence naming both | `test_lineage_seams.py::test_unify_contexts_raises_the_divergence_error_naming_both` |
| `vary-m48-A11` | §6.1d(B) identity arms | `test_lineage_seams.py::test_reindex_to_is_the_identity_for_a_value_already_at_the_target_or_context_free` |
| `vary-m48-A11` | §6.1d(B) descendant / divergent refusals | `test_lineage_seams.py::test_reindex_to_raises_when_the_values_handle_is_a_DESCENDANT_or_divergent` |
| `vary-m48-A11` | §6.1d link kind (1) with its result labels | `test_lineage_seams.py::test_link_kind_1_a_mask_derivation_makes_an_UNVARIED_value_varied` |
| `vary-m48-A11` | §6.1d link kind (2) with its result labels | `test_lineage_seams.py::test_link_kind_2_a_vary_link_is_the_IDENTITY_and_leaves_the_labels_alone` |
| `vary-m48-A11` | §6.1d link kind (3) with its result labels | `test_lineage_seams.py::test_link_kind_3_a_projection_link_returns_an_UNVARIED_value_with_no_labels` |
| `vary-m48-A12` | §4.1 one payload content hash across labels | `test_correctionlib_multiparam.py::test_all_labels_share_one_payload_content_hash` |
| `vary-m48-A12` | §4.1 labels differ ONLY in `systematic` | `test_correctionlib_multiparam.py::test_the_labels_differ_only_in_the_systematic_param` |
| `vary-m48-A12` | §1.2 only real content differences fork identity | `test_correctionlib_multiparam.py::test_the_three_universes_are_distinct_nodes_over_one_input` |
| — | shared context/awkward fixtures | `vary_ctx_fixtures.py` |
