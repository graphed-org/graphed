# awkward/m57 — the weight form's nominal names the factor being varied (traceability)

Milestone m57 (weight dedupe). Authority: the m57 plan §2 (contract), §4 (the properties, one test
each), §1 (the gap cases) and §3 (vocabulary only). The ambient weight is a registration-ordered list
of OPERATIONS: a *factor* multiplies its two-level member at the label; an *overlay* — a family whose
nominal named the whole ambient — replaces the running product at the labels its family covers. A
registration's nominal is compared BY NODE against the compositions the lineage has read and then
against every live factor's nominal, so the same weight is never multiplied in twice.

Run: `python -m pytest tests/frozen/awkward/m57 -q` (its own process, per the awkward per-milestone
split). The `m57_` helper prefix is load-bearing under prepend import mode. Every m57-new outcome —
the join, the overlay, the refusals, `ambient_entries`, `explain` — is reached only inside test
bodies, so the tree COLLECTS against a pre-m57 tree and fails at RUN time.

## Fixture — `m57_dedupe_fixtures.py`

* `m57_base()` / `m57_shifted(ctx, name, scale)` / `m57_weight(ctx, name, central, members)` /
  `m57_delta(ctx, name, handle)` — a root context carrying `Jet` and `MET`, a pT shift of both, a
  weight registration, and the relative-delta idiom (an ambient read as the nominal).
* `m57_sf(jets, table)` — the b-tag SF: one step per jet above `PT_CUT`, scaled by the systematic
  table. `m57_pu(jets)` is a pure weight off the jet MULTIPLICITY (no pT shift moves it);
  `m57_trig(met)` is a second pure weight. Every per-event value is a DYADIC rational with a small
  denominator, so every product of operations is exact whatever association the composition picks and
  the oracle compares bit-for-bit.
* The oracle: `m57_factor(name, nominal, members)` / `m57_overlay(name, members)` build the expected
  operations, `m57_reindexed(op, ctx)` and `m57_joined(op, name, members)` are what a row-space change
  and a join do to one, and `m57_ambient(ops, label)` folds them — a factor multiplies its member at
  the label, an overlay replaces the running product where it declares one. `m57_at(value, label)` is
  the two-level read the composition makes.
* `m57_two_factors()` is the two-live-factor base (`pu`, then `hf` over the SF central); `m57_tour()`
  is §1's chain — `pu`, `hf` on `sf`, the overlay `mu` read over both, `lf` on the same central behind
  the overlay, and `lf`'s diagonal placement at `{hf: up, mu: up}`.
* Instruments: `m57_node_count` (the store's node count), `m57_OpSpy` (calls to `Session.record_op`,
  the cost-class leg's live counter), `m57_ir` (the serialized IR of every universe),
  `m57_ambient_nodes` / `m57_ambient_values`.
* `m57_explain(ctx)` and `m57_entries(ctx)` import `graphed.systematics` INSIDE the call, so the legs
  that need them collect here and raise at run time until the package exists.

## Traceability (plan §4 bullet → test)

| Plan bullet (§4) | Test |
|---|---|
| two families on one central: every universe the one-SF oracle, nominal the central node | `test_join.py::test_two_families_on_one_central_give_the_one_sf_oracle_and_the_central_node` |
| two centrals sharing a nominal node with different shift coordinates, both orders | `test_join.py::test_two_centrals_with_different_shift_coordinates_agree_in_both_orders` |
| one coordinate declared by both with different nodes: refused before anything is minted | `test_join.py::test_a_coordinate_both_centrals_declare_with_different_nodes_is_refused_before_minting` |
| the m56 capstone shape with a weight the shift MOVES, both orders | `test_join.py::test_the_capstone_mints_the_same_labels_and_values_in_both_registration_orders` |
| a joint of two families on one factor is the container's cross member | `test_join.py::test_a_joint_of_two_families_on_one_factor_is_the_containers_cross_member` |
| the ratio spelling whose placed joint carries both variations | `test_join.py::test_the_ratio_spellings_placed_joint_carries_both_families_variations` |
| a family extending a factor still fans out over the shift it reads | `test_join.py::test_a_family_extending_a_factor_still_fans_out_over_the_shift_it_reads` |
| node identity, not value: a recomputed central is a new factor (control) | `test_join.py::test_a_recomputed_central_with_equal_values_is_a_new_factor` |
| name identity extends: a `jes` weight on `hf`'s central | `test_join.py::test_name_identity_extends_the_factor_the_central_names` |
| the ambient as nominal: the overlay's member node, everything earlier untouched | `test_overlay.py::test_the_ambient_as_nominal_leaves_every_earlier_universe_untouched` |
| §1(e) the two idioms in sequence: a join and a new factor around the overlay | `test_overlay.py::test_a_join_and_a_new_factor_around_the_overlay_give_the_one_sf_oracle` |
| relative-delta members over another family's labels compose (no joint) | `test_overlay.py::test_relative_delta_members_over_another_familys_labels_compose` |
| a handle read before a join decides and values as one read after it | `test_reads.py::test_a_handle_read_before_a_join_decides_and_values_as_one_read_after_it` |
| a handle read over a prefix anchors after that prefix | `test_reads.py::test_a_handle_read_over_a_prefix_anchors_after_that_prefix` |
| two relative-delta families agree in both orders, read spellings and prefix lengths | `test_reads.py::test_two_relative_delta_families_agree_in_both_orders_and_prefix_lengths` |
| an overlay insertion leaves a whole-list handle FRESH | `test_reads.py::test_an_overlay_insertion_leaves_a_whole_list_handle_fresh` |
| the adoption's own read anchors the overlay after the head at a masked child | `test_reads.py::test_at_a_masked_child_the_adopted_read_anchors_the_overlay_after_the_head` |
| at `graphed.nominal(ctx)` a child factor makes both orders agree | `test_reads.py::test_at_the_nominal_projection_a_child_factor_makes_both_orders_agree` |
| without the child factor the label sets differ while shared values agree (control) | `test_reads.py::test_without_a_child_factor_the_orders_label_sets_differ_but_every_shared_value_agrees` |
| an ancestor-factor join AFTER the overlay leaves its universe unchanged | `test_reads.py::test_an_ancestor_factor_join_after_the_overlay_leaves_its_universe_unchanged` |
| naming a factor that does not own `L` joins it and keeps the universe | `test_projection.py::test_naming_a_factor_that_does_not_own_the_label_joins_it_and_keeps_the_universe` |
| naming the factor that OWNS `L` is refused | `test_projection.py::test_naming_the_factor_that_owns_the_label_is_refused` |
| at a placed label both owners are refused and a third factor joins | `test_projection.py::test_at_a_placed_label_both_owners_are_refused_and_a_third_factor_joins` |
| the owner's recorded member is refused; a re-derivation below a mask is not (control) | `test_projection.py::test_a_central_equal_to_the_owners_member_is_refused_while_a_re_derivation_below_a_mask_is_not` |
| below a SECOND projection the owner of each universe is refused | `test_projection.py::test_below_a_second_projection_the_owner_of_each_universe_is_refused` |
| a node that is both a nominal and the owner's member joins that factor, either order | `test_projection.py::test_a_node_that_is_both_a_nominal_and_the_owners_member_joins_that_factor` |
| at a shift label the re-derived SF joins the shift-dependent factor | `test_projection.py::test_at_a_shift_label_the_re_derived_sf_joins_the_shift_dependent_factor` |
| at a shift label a join without that coordinate keeps the other factor's dependence | `test_projection.py::test_at_a_shift_label_a_join_without_that_coordinate_keeps_the_other_factors_dependence` |
| inside an overlay's universe: post-overlay join multiplies, covered join refused, head handle anchors | `test_projection.py::test_inside_an_overlays_universe_a_post_overlay_join_multiplies_and_a_covered_one_is_refused` |
| below a further projection the post-overlay join still keeps the overlay's factor | `test_projection.py::test_below_a_further_projection_the_post_overlay_join_keeps_the_overlays_factor` |
| a fixed overlay stays fixed through a further row-space link | `test_projection.py::test_a_fixed_overlay_stays_fixed_through_a_further_row_space_link` |
| `universe(weight(ctx), L)` built at the parent is an overlay after the head; above a mask a factor | `test_projection.py::test_a_parent_built_projected_nominal_is_an_overlay_after_the_head` |
| at the nominal projection a re-derived central joins; at a weight label it stays a factor | `test_projection.py::test_at_the_nominal_projection_a_re_derived_central_joins_while_a_weight_label_stays_a_factor` |
| one and two parent factors: a family on a masked child extends the ancestor's factor | `test_lineage.py::test_a_family_on_a_masked_child_extends_the_ancestors_factor` |
| the same central recomputed at the child is a new factor (control) | `test_lineage.py::test_the_same_central_recomputed_at_the_child_is_a_new_factor` |
| a bare `Varied({"nominal": container})` is refused with the pre-m57 error (control) | `test_lineage.py::test_a_bare_varied_over_a_live_factors_central_is_refused_as_before` |
| a crossed handle registers at the child beside an ancestor-factor join, both orders | `test_lineage.py::test_a_crossed_handle_registers_at_the_child_beside_an_ancestor_factor_join` |
| a crossed handle over a strict prefix anchors after that prefix at the child | `test_lineage.py::test_a_crossed_handle_over_a_strict_prefix_anchors_after_that_prefix_at_the_child` |
| a crossed handle over a factor a join widened is refused as stale | `test_lineage.py::test_a_crossed_handle_over_a_widened_factor_is_refused_as_stale` |
| two families on ONE ancestor central are one factor at the child | `test_lineage.py::test_two_families_on_one_ancestor_central_are_one_factor_at_the_child` |
| two families on TWO different ancestor factors extend each of them | `test_lineage.py::test_two_families_on_two_different_ancestor_factors_extend_each_of_them` |
| a central from the top of a two-mask chain names its factor at every depth | `test_lineage.py::test_a_central_from_the_top_of_a_two_mask_chain_names_its_factor_at_every_depth` |
| an ancestor-factor join after a crossed overlay agrees in both orders | `test_lineage.py::test_an_ancestor_factor_join_after_a_crossed_overlay_agrees_in_both_orders` |
| a union that widens a nominal member an overlay covers is refused | `test_staleness.py::test_a_union_that_widens_a_nominal_member_an_overlay_covers_is_refused` |
| a handle read before such a join is refused in either read spelling | `test_staleness.py::test_a_handle_read_before_such_a_join_is_refused_with_and_without_an_intervening_read` |
| the handle read AFTER the join gives the shifted joint | `test_staleness.py::test_the_handle_read_after_the_join_gives_the_shifted_joint` |
| transactional: a refusal after the match leaves the list and its generations untouched | `test_staleness.py::test_a_refusal_after_the_match_leaves_the_list_and_its_generations_untouched` |
| an overlay leaves the composition's cost class | `test_cost_class.py::test_an_overlay_leaves_the_compositions_cost_class` |
| a placement minted after an ambient read keeps its declared member | `test_placements.py::test_a_placement_minted_after_an_ambient_read_keeps_its_declared_member` |
| a placement at a point carrying the overlay's coordinate keeps its member, both prefixes | `test_placements.py::test_a_placement_at_a_point_carrying_the_overlays_coordinate_keeps_its_member` |
| the overlay family placing its OWN universe off its axis keeps that member | `test_placements.py::test_the_overlay_family_placing_its_own_universe_off_its_axis_keeps_that_member` |
| `variations` reports the extending family as `Kind.WEIGHT`; labels keep order (control) | `test_records.py::test_variations_reports_the_extending_family_as_a_weight_and_labels_keep_order` |
| the shift-after-weight diagnostic names a family that joined a factor (control) | `test_records.py::test_the_shift_after_weight_diagnostic_names_a_family_that_joined_a_factor` |
| the rider leg: entries carry the families, kind and links of the parents' | `test_records.py::test_ambient_entries_carry_the_families_kind_and_links_of_the_parents` |
| `ambient_entries` lists the order the composition applies | `test_records.py::test_ambient_entries_list_the_order_the_composition_applies` |
| the listing keeps its slots, kinds and order across a value-free join | `test_records.py::test_ambient_entries_keep_their_slots_kinds_and_order_across_a_value_free_join` |
| explain: each family's entry form and every label's origin | `test_explain.py::test_explain_reports_each_familys_entry_form_and_every_labels_origin` |
| explain: the "composes with" sets of the §1 chain | `test_explain.py::test_explain_reports_the_composes_with_sets_of_the_chain` |
| explain: a joined family SHARES the factor, never composes with its other family | `test_explain.py::test_explain_reports_a_joined_family_as_sharing_the_factor` |
| explain: the registering context's and the entries' row-space links | `test_explain.py::test_explain_reports_the_links_the_lineage_took` |
| explain: a fixed overlay is marked and its line names the universe that fixed it | `test_explain.py::test_explain_marks_a_fixed_overlay_and_names_the_universe_that_fixed_it` |
| explain: a family whose label the projection dropped keeps its relations and placements | `test_explain.py::test_explain_keeps_the_relations_of_a_family_whose_label_the_projection_dropped` |
| explain: the capstone's fan-out and the pure-weight family's independence | `test_explain.py::test_explain_reports_the_capstones_fanout_and_independence` |
| explain: a weight registered before the shift reads objects a later shift moved | `test_explain.py::test_explain_reports_a_weight_registered_before_the_shift_as_reading_moved_objects` |
| explain: the Session's node count after `explain` equals the count after a `weight` read | `test_explain.py::test_explain_mints_no_more_than_a_weight_read` |
| explain: the text is byte-identical across two Sessions whose ids differ | `test_explain.py::test_the_explain_text_is_byte_identical_across_two_sessions` |
| determinism: the extending program serializes byte-identically (control) | `test_determinism.py::test_the_extending_program_serializes_byte_identically_across_two_sessions` |
| no silent change: a program that names nothing mints exactly what it minted before (control) | `test_determinism.py::test_a_program_that_names_nothing_mints_exactly_the_composition_it_minted_before` |

## The surfaces these tests bind

`graphed.systematics` does not exist pre-m57, so `m57_explain` / `m57_entries` import it inside the
call and the legs that need it fail with `ModuleNotFoundError` until it does. The shape those legs
read is the plan's own vocabulary, and freezing them fixes it:

* `ambient_entries(ctx)` yields records unpackable as `(slot, rider, entry)`, in the order `_compose`
  applies them. A rider has `kind` (`"factor"` / `"overlay"`), `families` (the families it carries,
  keyed by name), `links` (the row-space links it came through, as `(kind, label)` pairs — `("mask",
  None)`, `("project", "hf_up")`) and `fixed`.
* `explain(ctx)` returns an `Explanation` with `families` (keyed by name), `operations` and
  `variations` (keyed by label), and a one-line-per-item `str()`. A family record has `links`,
  `entry` (with `kind` in `"factor"` / `"join"` / `"overlay"` / `"shift"` and the `families` it
  names), `placements` (its registered points) and the relation sets `composes_with`, `shares_with`,
  `fans_out_over` and `independent_of`. An `Operation` has `kind`, `families`, `links` and `fixed`,
  and its own `str()` line.

## Non-vacuity — what each group does on a pre-m57 tree

Against a pre-m57 tree the suite COLLECTS with zero errors and every test fails EXCEPT the eight
controls listed at the end of this section — and the same tests fail across two runs. Regenerate with
`python -m pytest tests/frozen/awkward/m57 -q -rp`.

* The join, overlay, read, lineage, cost-class and placement groups FAIL on the squared central or
  the squared ambient — the ambient's value and/or its per-label node ids move: e.g. two families on
  one SF give `SF²` at nominal (`[4, 64, …]` for `[2, 8, …]`), a relative-delta family multiplies the
  whole composition in again, and a family on a masked child multiplies the ancestor's central a
  second (and, with three families, a third) time.
* A refusal leg the test reaches fails with `DID NOT RAISE GraphedError`: pre-m57 nothing compares
  nodes, so the owner of a projected universe, the stale handle, the covered join and the conflicting
  coordinate union are all silently accepted as products. Where a refusal is a test's SECOND property,
  the value the join ahead of it must produce fails first.
* The cost-class leg fails on its VALUE half, not its cost half: pre-m57 the handle is itself a
  factor, so the work already sits in the tree's class. The cost bound and the node ceiling are the
  guards against an m57 build that re-multiplies the list per label (at sixteen factors such a walk
  leaves both).
* The `ambient_entries` and `explain` legs FAIL with `ModuleNotFoundError: No module named
  'graphed.systematics'`.
* The eight PASSING legs are the declared controls — each is a live instrument showing the harness is
  alive and the decision really is a decision:
  * `test_a_recomputed_central_with_equal_values_is_a_new_factor` and
    `test_the_same_central_recomputed_at_the_child_is_a_new_factor`: node identity, never value, so
    these stay products then and now.
  * `test_a_bare_varied_over_a_live_factors_central_is_refused_as_before`: the pre-m57 form check
    already refuses a central nested past one level.
  * `test_without_a_child_factor_the_orders_label_sets_differ_but_every_shared_value_agrees`: the
    label sets differ at a nominal projection then and now; the agreement of the shared values is the
    property.
  * `test_variations_reports_the_extending_family_as_a_weight_and_labels_keep_order` and
    `test_the_shift_after_weight_diagnostic_names_a_family_that_joined_a_factor`: the records the
    append path writes, which the join path must keep writing.
  * both `test_determinism.py` legs: the program is deterministic then and now, and a program that
    names nothing must keep minting exactly the composition it minted — pinned by rebuilding every
    universe's product from the test's own operands and showing the rebuild mints no node.
* Freshness has no public accessor, so it is witnessed by outcome: the whole-list handle of
  `test_an_overlay_insertion_leaves_a_whole_list_handle_fresh` is accepted and reads the composition
  it named, against the stale handles of `test_staleness.py` and
  `test_a_crossed_handle_over_a_widened_factor_is_refused_as_stale`, which are refused.
