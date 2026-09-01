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
| `vary-m48-A1` | §1.1 `variations=` channel; PDF indices untouched | `test_tag_grammar.py::test_the_variations_channel_carries_tags_kwarg_syntax_cannot_spell` |
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
| `vary-m48-A1` | §2.1 `variations=` refused in the shift form | `test_tag_grammar.py::test_variations_is_refused_in_the_shift_form` |
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
| — | shared context/awkward fixtures | `vary_ctx_fixtures.py` |
