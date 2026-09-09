# awkward/m56 — both-kind fan-out (traceability)

Milestone m56 (fan-out decided on nodes: a nuisance that is both a shift and a weight). Authority:
`both-kind-fanout-plan.md` §2 (contract) and §4 (the properties, one test each). Composition is a
fact about NODES: a member's foreign coordinate is composition only where the member's node reads a
registered factor's VARIED member at that label, so a nuisance that shifts the objects AND swaps an
SF table by name identity (`Kind.WEIGHT | Kind.SHIFT`) is still a dependency when a second family
reaches it through the shifted jets.

Run: `python -m pytest tests/frozen/awkward/m56 -q` (its own process, per the awkward per-milestone
split). The `m56_` prefix is load-bearing under prepend import mode. Every m56-new outcome is
reached only inside test bodies, so the tree COLLECTS against a pre-m56 tree and fails at RUN time.

Fixtures — `m56_fanout_fixtures.py`:
* `m56_capstone(...)` — the headline: `jes` shifts `Jet`/`MET` and swaps the SF table by name
  identity; `jer` shifts them from the NOMINAL jets (shift-only, so no three-coordinate labels);
  `hf` is a weight family over the jets both vary. `weight_first=` flips the registration order;
  `placements=` and extra keywords land on the `hf` call. `jets` maps each label to the jet record
  that universe reads — the oracle's operands.
* `m56_both_kind(ctx)` / `m56_weight_family(ctx, name, member)` — the one-nuisance base and the
  three-member weight registration every probe leg uses.
* `m56_incidental(leg)` / `m56_cut_child(cut)` — the inclusion legs: a member multiplying the seed
  weight or an unrelated family's central, and a family registered on a cut over the seed weight
  (control: a kinematic cut). Each carries a bare `SF(varied jets)` sibling.
* `m56_two_both_kind()` / `m56_pure_weight()` / `m56_masked_child()` / `m56_spectator()` — the
  exclusion programs.
* `m56_joints(name, foreign)` — the four joints `name`(up/down) x `foreign`(up/down);
  `m56_minted(weight, name)` — the joints `name`'s registration actually minted.

| Plan bullet (§4) | Test |
|---|---|
| hf fans out over the both-kind nuisance as over the shift-only one; each joint is its own node | `test_both_kind_fanout.py::test_a_both_kind_nuisance_fans_out_beside_the_shift_only_one` |
| inclusion: the member multiplies the context's seed weight | `test_incidental_reads.py::test_a_member_multiplying_the_seed_weight_still_fans_out` |
| inclusion: the member multiplies an unrelated weight family's central | `test_incidental_reads.py::test_a_member_multiplying_another_familys_central_still_fans_out` |
| inclusion: registered on a cut over the seed weight (control: a kinematic cut) | `test_incidental_reads.py::test_a_family_on_a_cut_over_the_seed_weight_fans_out_as_on_a_kinematic_cut` |
| item 2 order independence: same universes, same Jet/MET/weight values | `test_order_independence.py::test_the_two_registration_orders_give_the_same_universes_and_values` |
| item 3 oracle: the two-level product in registration order, joint and one-at-a-time | `test_joint_oracle.py::test_the_joint_weight_is_the_two_level_product_of_the_factors` |
| exclusion: a member computed from the ambient, one-at-a-time and joint labels alike | `test_exclusions.py::test_a_member_computed_from_the_ambient_mints_nothing_under_two_both_kind_families` |
| exclusion: a member reading both the shifted objects and the ambient | `test_exclusions.py::test_a_member_reading_both_the_shifted_objects_and_the_ambient_mints_nothing` |
| exclusion: a pure-weight coordinate composed, the carried shift still fanned out | `test_exclusions.py::test_a_pure_weight_coordinate_is_composed_while_the_carried_shift_still_fans_out` |
| exclusion: a member built at the PARENT, registered on the mask-derived child | `test_exclusions.py::test_a_member_built_at_the_parent_stays_composed_on_the_masked_child` |
| exclusion: a spectator coordinate collapses | `test_exclusions.py::test_a_spectator_coordinate_collapses_while_the_carried_shift_fans_out` |
| item 5: `composes_as_union=True` collapses the new joints | `test_downstream_knobs.py::test_composes_as_union_collapses_the_cross_kind_joints` |
| item 5: a placement keeps a named cross-kind joint and prunes the rest | `test_downstream_knobs.py::test_a_placement_keeps_a_named_cross_kind_joint_and_prunes_the_rest` |
| item 5: the guard's bound is the product over EVERY foreign family | `test_downstream_knobs.py::test_the_guard_bounds_the_grid_over_every_foreign_family` |

## Non-vacuity — what each test does on a pre-m56 tree, and why

Pre-m56 the capstone mints 11 universes (`hf` x `jer` only); the four `hf` x `jes` joints are the
gap. The `jes` coordinate is dropped because `jes` is a family the ambient registers as a weight,
whatever the member's dataflow.

* headline / order independence / union / oracle — FAIL: `m56_minted(weight, "hf")` is the four
  `hf__jer` joints, not eight; the oracle leg reaches `hf_up__jes_up`, which resolves to the wrong
  universe and mismatches the hand-built product.
* placement — FAIL (`PointError`): `{"hf": "up", "jes": "up"}` "names no joint the fanout of 'hf'
  derives".
* guard — FAIL (`DID NOT RAISE`): the pre-m56 bound is `jer(3) x hf(3)` = 9, so a budget of 9 is
  admitted; the design's bound is `jer(3) x jes(3) x hf(3)` = 27 and must refuse it.
* the three inclusion legs — FAIL: both the probe family AND its bare `SF(varied jets)` sibling mint
  the empty set instead of the four `__jes` joints.
* the ambient / mixed / parent-ambient / spectator exclusion legs — the EXCLUSION half already holds
  pre-m56 (nothing minted), so each fails on its live instrument: the sibling that must mint. An
  empty set from these legs is therefore a decision, not a dead fixture.
* `test_a_pure_weight_coordinate_is_composed_while_the_carried_shift_still_fans_out` PASSES pre-m56
  — the declared positive control. A pure-weight family's coordinate is composition then and now,
  and the plain shift the member also carries fans out then and now (the m53 answer m56 preserves).
  A run in which it also failed would prove the harness dead rather than the fan-out absent.
