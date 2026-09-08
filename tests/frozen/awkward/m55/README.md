# awkward/m55 — lockstep by propagation (traceability)

Milestone m55: the shift form of `graphed.vary` accepts a `Varied` as a collection value, so a
collection that is a FUNCTION of another varied collection — Type-1 MET of the varied jets — moves
in lockstep by ordinary propagation instead of a hand-built `{tag: record}` map. Authority:
`graphed-workdir/lockstep-varied-plan.md` §2 (contract) and §4 (the properties below).

Run: `python -m pytest tests/frozen/awkward/m55 -q` (its own process, per the awkward per-milestone
split). The `m55_` helper prefix is load-bearing under prepend import mode. The m55 spelling is
reached only inside test bodies, so the tree COLLECTS against a tree with no m55 implementation and
fails at RUN time.

The toy: a jagged `Jet` record and `met_of(jets, raw_met)`, the plan's stand-in for Type-1 MET. The
context carries the PROPAGATED MET as its own central member, which is what makes a propagated
container's nominal intern to the context's MET node (§2 item 3) — the shape the tour capstone
builds. `Muon` and `RawMET` stay on the record, giving the tree a collection the context READS
instead of carries.

The equivalence witness throughout is node identity against the same program spelled by hand
(`{tag: graphed.member_of(container, f"{name}_{tag}")}`) in its own fresh Session; `equivalence()`
runs both and compares `{(collection, label): node id}`. Refusals go through `refused()`, which
requires a `GraphedError` and checks §2 item 5 on the spot: the Session's point registry and its
inverse are exactly what they were.

| Contract (§2) · §4 bullet | Test |
|---|---|
| item 1 shape, item 4 equivalence · b1 | `test_lockstep_equivalence.py::test_a_varied_jet_and_its_propagated_met_mint_the_hand_forms_nodes` |
| item 4 values · b1 | `test_lockstep_equivalence.py::test_the_jes_up_met_is_the_met_of_the_jes_up_jets` |
| item 1 mixing the two shapes · b2 | `test_lockstep_equivalence.py::test_a_tag_mapping_and_a_varied_mix_in_one_call` |
| item 1 lockstep over the unpacked tags · b2 | `test_lockstep_equivalence.py::test_a_varied_whose_tags_differ_from_the_mapping_is_out_of_lockstep` |
| item 1 two shapes and no third | `test_lockstep_equivalence.py::test_a_value_that_is_neither_a_mapping_nor_a_varied_stays_refused` (control) |
| item 2 inherited labels · b3 | `test_family_purity.py::test_a_container_inheriting_another_familys_labels_is_refused` |
| item 2 joint labels · b3 | `test_family_purity.py::test_a_container_whose_members_read_a_varied_collection_is_refused` |
| item 2 another family · b3 | `test_family_purity.py::test_a_container_varying_a_different_family_is_refused` |
| item 2 paired accept · b3 | `test_family_purity.py::test_the_same_second_family_built_on_the_nominal_is_accepted` |
| item 2 stacking a new tag · b4 | `test_family_stacking.py::test_a_new_tag_on_the_nominal_stacks_onto_the_registered_family` |
| item 2 re-offered tag, `check_family` · b4 | `test_family_stacking.py::test_a_container_re_offering_a_registered_tag_is_refused` |
| item 3 accept across a mask link · b5 | `test_lineage_nominal.py::test_a_parent_built_container_registers_at_a_masked_child` |
| item 3 accept across a vary link · b5 | `test_lineage_nominal.py::test_a_parent_built_container_registers_at_a_vary_link_descendant` |
| item 3 refusal, nominal moved · b5 | `test_lineage_nominal.py::test_a_container_whose_nominal_was_rescaled_is_refused_naming_both_nodes` |
| item 3 refusal, projected child · b5 | `test_lineage_nominal.py::test_a_parent_built_container_is_refused_at_a_projected_child` |
| item 3 refusal, record-read collection · b5 | `test_lineage_nominal.py::test_a_record_read_collection_is_refused_at_a_masked_child_and_accepted_at_its_own` |
| item 1 placement channel · b6 | `test_points_channels.py::test_a_placement_beside_a_varied_member_is_refused_pointing_at_the_hand_form` |
| item 1 declaring channel · b6 | `test_points_channels.py::test_the_declaring_points_channel_stays_refused_in_the_shift_form` (control) |
| item 5 transactional · b7 | `test_transactional_registry.py::test_a_refused_call_leaves_the_family_registrable` |
| item 6 loose-form re-mint | every accept: each program registers the family loosely to build the container, then again in the shift call. A registry that refused the identical re-mint fails every accept. |

## Non-vacuity (the pre-m55 tree, run twice, identical failing set)

19 tests: 17 fail at RUN time with `collection '<name>' needs a {tag: record} mapping, got Varied`
— the pre-m55 refusal — and 2 pass. The two that pass are the declared regression controls, the
live-harness positive control: a run in which they also failed would prove the harness dead rather
than the feature absent.

What defect makes each group fail once the feature exists:

* the accepts (`equivalence`) — a container unpacked to the wrong members, or to the nominal for
  every label. The distinctness assertion beside each is what a collapse trips; the node maps being
  equal is what a wrong member trips.
* `test_the_jes_up_met_is_the_met_of_the_jes_up_jets` — the MET universe carrying an array that is
  not the propagated one; the second assertion refuses a MET that never moved.
* item 2's three refusals — a rule that admits a container carrying a foreign label, and each
  message assertion names a fragment (collection, family, the extra labels, the hand-form pointer)
  a bare `GraphedError` would not carry.
* item 3's three refusals — an admission rule keyed on where the container was BUILT instead of on
  the node the reindexed nominal lands on. Each asserts both node ids appear in the message, and the
  last two additionally assert the hand form still registers the same program, so a check placed in
  the shared path fails there.
* `test_a_placement_beside_a_varied_member_is_refused_pointing_at_the_hand_form` — the same call
  without the placement is asserted accepted, so the refusal must come from the placement.
* the transactional test — a refusal that mints before it raises leaves the family unregisterable
  for the life of the Session; the retry is what catches it.

## Not pinned here

§2 item 1 widens two existing messages (the declaring `points=` refusal, and `_check_lockstep`'s
"needs a `{tag: record}` mapping") to name a `Varied` as a second accepted collection value. Both
fire only where m55 changes no behaviour, so a test of the new wording would fail on the pre-m55
tree for a message reason rather than the pre-m55 refusal, which the sanity discipline reserves for
behaviour. The two controls above pin what those messages must keep saying.
